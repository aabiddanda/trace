"""Cython-based helper functions for HMM implementation for ghost admixture."""
from libc.math cimport erf, exp, lgamma, log
from libcpp cimport bool

from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.stats import binom, hypergeom

from libc.stdio cimport printf
from libc.stdlib cimport RAND_MAX, rand, srand

import numpy as np

cimport cython
cimport numpy as cnp

cnp.import_array()

DTYPE_64 = np.int64
DTYPE_32 = np.int32
DTYPE_8 = np.int8
DTYPE_64_f = np.float64
UINT_8 = np.uint8
UINT_16 = np.uint16
UINT_32 = np.uint32
UINT_64 = np.uint64
ctypedef cnp.int64_t DTYPE_64_t
ctypedef cnp.int32_t DTYPE_32_t
ctypedef cnp.int8_t DTYPE_8_t
ctypedef cnp.float64_t DTYPE_64_f_t
ctypedef fused UINT:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] create_Q_cython(int n):
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] Q = np.zeros((n, n))
  cdef double [:, ::1] Q_view = Q
  cdef Py_ssize_t i
  cdef int x
  for i, x in enumerate(range(n, 1, -1)):
    Q_view[i, i] = -x * (x - 1) / 2
    Q_view[i, i + 1] = x * (x - 1) / 2
  return Q

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double exp_focal_coal_interval_cython(int n, double t):
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] Q = create_Q_cython(n)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] P1 = expm(t * Q)
  cdef double tot_expectation = 0.0
  cdef double [:, ::1] P1_view = P1
  cdef double ps = 0.0
  cdef Py_ssize_t nt0
  cdef int j
  for nt0 in range(1, n):
    ps = 0.0
    for j in range(n, nt0, -1):
      ps += 2.0 / j
    tot_expectation += ps * P1_view[0, n - nt0]
  return tot_expectation

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] exp_ncoal_null(long [::1] nt1, long [::1] nt2):
  """Compute the expected number of coalescent events under the null."""
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] output = np.zeros(nt1.shape[0])
  cdef double [:] output_view = output
  cdef Py_ssize_t i
  cdef int j
  cdef double y = 1e-9;
  for i in range(nt1.shape[0]):
    for j in range(nt2[i] + 1, nt1[i] + 1):
      output_view[i] += 2.0 / j
  return output

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double exp_ncoal_intro(long nt1, double ne, double t_interval, double padmix, double ne_archaic = 10000.0):
  """Compute the expected number of coalescent events under the introgression model."""
  cdef double output = 0.0
  cdef int j
  cdef double tot
  for j in range(2, nt1 + 1):
    p = hypergeom.pmf(j, 2 * ne, int(2 * ne * padmix), nt1) * j / (padmix * nt1)
    output += exp_focal_coal_interval_cython(j, t_interval / (2 * ne_archaic)) * p
  return output

cdef double xlogy(double x, double y):
  """Implementation of x*log(y)"""
  if x == 0.0:
    return 0.0
  else:
    return x * log(y)

cdef double poisson_logpmf(double x, double mu):
  """Return the logpmf of the Poisson distribution."""
  return xlogy(x, mu) - mu - lgamma(x + 1)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] extract_n_coal_cython(double [:, ::1] ncoal, double tadmix, double tarchaic):
  """Extract the number of coalescent events from the matrix."""
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] output = np.zeros(ncoal.shape[0])
  cdef double [::1] output_view = output
  cdef Py_ssize_t i, j
  for i in range(ncoal.shape[0]):
    for j in range(ncoal.shape[1]):
      if ncoal[i, j] > tadmix and ncoal[i, j] < tarchaic:
        output_view[i] += 1
  return output

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] emission_coal_event(double [::1] x, int z, double [::1] lamb_n, double lamb0):
  """Emission model for number of coalescent events.

  Arguments:
    x: number of coalescent events on branch in time interval, np array
    z: binary indicator of null or ghost admixture state
    lamb_null: expected number of coalescent events under the null
    lamb0: expected wiggle room for reduced portion of lambda

  Returns:
    emission probability of a specific outcome

  """
  cdef output = np.zeros(x.size)
  cdef double [::1] output_view = output
  cdef Py_ssize_t i

  if z == 0:
    # The null state (note this still uses the poisson PMF here)
    for i in range(len(x)):
      output_view[i] = poisson_logpmf(x[i], lamb_n[i])
  else:
    for i in range(len(x)):
      output_view[i] = poisson_logpmf(x[i], lamb0)
  return output

cdef double gamma_logpdf(double x, double a, double b):
  """The logpdf of the gamma distribution."""
  return (a - 1)*log(x) - b*x + xlogy(a,b) - lgamma(a)

cdef double gamma_pdf(double x, double a, double b):
    """The PDF of gamma distribution"""
    return (b**a * x**(a-1) * exp(-b*x)) / exp(lgamma(a))

cdef double logsumexp(double[:] x):
  """Custom definition of the logsumexp function for optimization."""
  cdef int i,n;
  cdef double m = -1e32;
  cdef double c = 0.0;
  n = x.size
  for i in range(n):
      m = max(m,x[i])
  for i in range(n):
      c += exp(x[i] - m)
  return m + log(c)

cdef double logaddexp(double a, double b):
  """Simple logaddexp function for just two numbers."""
  cdef double m = -1e32;
  cdef double c = 0.0;
  m = max(a,b)
  c = exp(a - m) + exp(b - m)
  return m + log(c)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] emission_product_z1(double[:, ::1] x, double alpha, double a1, double b1, double a2, double b2):
  cdef double logl = 0.0
  cdef int i, j, m, t
  m = x.shape[1]
  t = x.shape[0]
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] em = np.zeros(t)
  cdef double [::1] em_view = em
  for j in range(t):
    logl = 0.0
    for i in range(m):
      logl += logaddexp(log(alpha) + gamma_logpdf(x[j, i], a1,b1),  log((1.0 - alpha)) + gamma_logpdf(x[j, i], a2, b2))
    em_view[j] = logl / m
  return em

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] emission_product_z1_flat(double[:, ::1] x, double alpha, double a1, double b1, double a2, double b2, double a, double x_prime, double integrand, double interval):
  cdef double logl = 0.0;
  cdef int i, j, m, t
  m = x.shape[1]
  t = x.shape[0]
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] em = np.zeros(t)
  cdef double [::1] em_view = em
  for j in range(t):
    logl = 0.0
    for i in range(m):
      if x[j, i] < x_prime:
        logl +=  log((1 - a)/integrand) + logaddexp(log(alpha) + gamma_logpdf(x[j, i], a1,b1),  log((1.0 - alpha)) + gamma_logpdf(x[j, i], a2, b2))
      else:
        logl += log(a / interval)
    em_view[j] = logl / m
  return em

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef forward_algo_product(double[:, ::1] es , double p=1e-2, double q=1e-2, double pi0=0.5):
  """Cython implementation of a helper function for the forward algorithm.

  Arguments:
    es: emission probabilities (log space)
    p: transition probability from state 0 to state 0
    q: transition probability from state 1 to state 1
    pi0: initial probability of being in state 0
  """
  cdef int i, j, n, m
  cdef float p_i, q_i, cur_emission0, cur_emission1
  n = es.shape[1]
  m = 2
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] alphas = np.zeros(shape=(m, n))
  cdef double [:, ::1] alphas_view = alphas
  alphas_view[0, 0] = log(pi0) + es[0, 0]
  alphas_view[1, 0] = log(1 - pi0) + es[1, 0]
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] scaler = np.zeros(n)
  cdef double [::1] scaler_view = scaler
  scaler_view[0] = logsumexp(alphas_view[:, 0])
  for i in range(m):
    alphas_view[i, 0] = alphas_view[i, 0] - scaler_view[0]
  for i in range(1, n):
    p_i = p
    q_i = q
    # This is in log-space ...
    cur_emission0 = es[0, i]
    cur_emission1 = es[1, i]
    alphas_view[0, i] = cur_emission0 + logaddexp(log(1 - p_i) + alphas_view[0, (i - 1)], log(q_i) + alphas_view[1, (i - 1)])
    alphas_view[1, i] = cur_emission1 + logaddexp(log(p_i) + alphas_view[0, (i - 1)], log(1 - q_i) + alphas_view[1, (i - 1)])
    scaler_view[i] = logsumexp(alphas_view[:, i])
    for j in range(m):
      alphas_view[j, i] = alphas_view[j, i] - scaler_view[i]
  # Returns the alphas, scaler, and sum of the scaler (or loglik)
  return alphas, scaler, sum(scaler)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef backward_algo_product(double[:, ::1] es, double p=1e-2, double q=1e-2):
  """Cython implementation of backward algorithm.

  Arguments:
    es: emission probabilities (log space)
    p: transition probability from state 0 to state 0
    q: transition probability from state 1 to state 1
  """
  cdef int i, j, n, m
  cdef float p_i, q_i, cur_emission0, cur_emission1
  n = es.shape[1]
  m = 2
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=2, mode="c"] betas = np.zeros(shape=(m, n))
  cdef double [:, ::1] betas_view = betas
  betas_view[0, n - 1] = 0
  betas_view[1, n - 1] = 0
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] scaler = np.zeros(n)
  cdef double [::1] scaler_view = scaler
  scaler_view[n - 1] = logsumexp(betas_view[:, n - 1])
  for i in range(m):
    betas_view[i, n - 1] = betas_view[i, n - 1] - scaler_view[n - 1]
  for i in range(n - 2, -1, -1):
    p_i = p
    q_i = q
    # Calculate the full set of emissions
    cur_emission0 = es[0, i + 1]
    cur_emission1 = es[1, i + 1]
    # betas(z_i) = sum_{j}(beta(z_{i+1}) * p(x_{i+1} | z_{i+1}) * p(z_{i+1} | z_i))
    betas_view[0, i] = logaddexp(betas_view[0, (i + 1)] + cur_emission0 + log(1 - p_i), betas_view[1, (i + 1)] + cur_emission1 + log(p_i))
    betas_view[1, i] = logaddexp(betas_view[0, (i + 1)] + cur_emission0 + log(q_i), betas_view[1, (i + 1)] + cur_emission1 + log(1 - q_i))
    # Do the rescaling here ...
    scaler_view[i] = logsumexp(betas_view[:, i])
    for j in range(m):
      betas_view[j, i] = betas_view[j, i] - scaler_view[i]
  return betas, scaler, sum(scaler)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef update_oneind_cython(int m, double [:, ::1] alphas, double [:, ::1] betas, double [:, ::1] emissions, double p, double q):
  """Update the transition probabilities for the one-individual case.

  Arguments:
    m: number of trees
  """
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] n = np.zeros(m - 1)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] eta01 = np.zeros(m - 1)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] eta10 = np.zeros(m - 1)
  cdef double[4] norm_factor
  cdef double [::1] n_view = n
  cdef double [::1] eta01_view = eta01
  cdef double [::1] eta10_view = eta10
  cdef int i, j

  for i, j in zip(range(m - 1), range(1, m)):
      norm_factor = [0.0, 0.0, 0.0, 0.0]
      norm_factor[0] = (
          alphas[0, i]
          + log(1.0 - p)
          + betas[0, j]
          + emissions[0, j]
      )
      norm_factor[1] = (
          alphas[0, i]
          + log(p)
          + betas[1, j]
          + emissions[1, j]
      )
      norm_factor[2] = (
          alphas[1, i]
          + log(q)
          + betas[0, j]
          + emissions[0, j]
      )
      norm_factor[3] = (
          alphas[1, i]
          + log(1 - q)
          + betas[1, j]
          + emissions[1, j]
      )
      n_view[i] = logsumexp(norm_factor)
      eta01_view[i] = (
          alphas[0, i]
          + log(p)
          + betas[1, j]
          + emissions[1, j]
      ) - n_view[i]
      eta10_view[i] = (
          alphas[1, i]
          + log(q)
          + betas[0, j]
          + emissions[0, j]
      ) - n_view[i]
  return eta01, eta10

cdef double posterior_assignment(double x, double alpha, double a1, double a2, double b1, double b2):
    """For a given outcome - assign a specific posterior threshold to it."""
    cdef double z, log_f_tot, log_f_num;
    log_f_tot = logaddexp(log(alpha) + gamma_logpdf(x, a1, b1), log(1.0 - alpha) + gamma_logpdf(x, a2, b2))
    log_f_num = log(alpha) + gamma_logpdf(x, a1, b1)
    z = exp(log_f_num - log_f_tot)
    if not z > 0: # numerical stability
        z = 1e-5
    return z

def ecm_update_shape(xs, zs0, zs1, b0, b1):
    """Expect conditional maximization for shape parameter in a non-shared model.

    Arguments:
      xs (`np.array`): numpy array of x values
      zs0 (`np.array`): probability of being in the 0 component

    NOTE: this is based on Derek S. Young et al 2019
        Finite mixture-of-gamma distributions: estimation, inference, and model-based clustering Eq 3
    """
    f0 = lambda a: np.sum(zs0 * (np.log(xs) + np.log(b0) - digamma(a))) #noqa
    f1 = lambda a: np.sum(zs1 * (np.log(xs) + np.log(b1) - digamma(a))) #noqa
    alpha_hat = np.sum(zs0) / zs0.size
    a0_hat = brentq(f0, 1e-6, 1e6)
    a1_hat = brentq(f1, 1e-6, 1e6)
    return a0_hat, a1_hat, alpha_hat

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (double, double) ecm_update_scales(cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] xs, cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] zs0, cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] zs1, double a0, double a1):
    """Expect conditional maximization for scale parameters.

    Arguments:
      - xs (`np.array`): array of flattened tmrcas.
      - zs0 (`np.array`): array of inclusion probabilities in first component.
      - zs1 (`np.array`): array of inclusion probabilities in second component.
      - a0 (`float`): shape parameter of first component.
      - a1 (`float`): shape parameter of second component.
    """
    cdef double b0_hat = a0 * np.sum(zs0) / np.sum(xs * zs0)
    cdef double b1_hat = a1 * np.sum(zs1) / np.sum(xs * zs1)
    return b0_hat, b1_hat

def ecm_full_update(xs_flat, alpha, a1, a2, b1, b2):
  """Compute the two ECM updates in lockstep.

  Arguments:
    - xs_flat (`float`): all tmrca values used for mix gamma estimation
    - alpha (`float`): the proportion in the first tmrca cutoff
    - a1 (`float`): the shape parameter for t_admix
    - a2 (`float`): the shape parameter for t_archaic
    - b1 (`float`): the scale parameter for t_admix
    - b2 (`float`): the scale parameter for t_archaic
  """
  if xs_flat.size > 0:
    zs0 = np.array([posterior_assignment(x, alpha, a1, a2, b1, b2) for x in xs_flat])
    a1_hat, a2_hat, alpha_hat = ecm_update_shape(xs_flat, zs0, 1 - zs0, b1, b2)
    b1_hat, b2_hat = ecm_update_scales(xs_flat, zs0, 1 - zs0, a1_hat, a2_hat)
    if alpha_hat > 0 and a1_hat > 0 and a2_hat > 0 and b1_hat > 0 and b2_hat > 0:
      return alpha_hat, a1_hat, a2_hat, b1_hat, b2_hat
  return alpha, a1, a2, b1, b2
