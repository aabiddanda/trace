""""Cython-based helper functions for HMM implementation for ghost admixture."""
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
cpdef double extract_n_coal_cython(double [::1] ncoal, double tadmix, double tarchaic):
  """Extract the number of coalescent events from the matrix."""
  cdef double output = 0
  cdef Py_ssize_t i, j
  for i in range(len(ncoal)):
    if ncoal[i] > tadmix and ncoal[i] < tarchaic:
      output += 1
  return output

cpdef double gamma_logpdf(double x, double a, double b):
  """The logpdf of the gamma distribution."""
  return (a - 1)*log(x) - b*x + xlogy(a,b) - lgamma(a)

cdef double gamma_pdf(double x, double a, double b):
    """The PDF of gamma distribution"""
    return (b**a * x**(a-1) * exp(-b*x)) / exp(lgamma(a))

cpdef double logsumexp(double[:] x):
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
cpdef update_oneind_cython(double [:, ::1] alphas, double [:, ::1] betas, double [:, ::1] emissions, double p, double q):
  """Update the transition probabilities for the one-individual case.

  Arguments:
    m: number of trees
  """
  cdef int m = alphas.shape[1]
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] n = np.zeros(m - 1)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] eta01 = np.zeros(m - 1)
  cdef cnp.ndarray[DTYPE_64_f_t, ndim=1, mode="c"] eta10 = np.zeros(m - 1)
  cdef double[4] norm_factor
  cdef double [::1] n_view = n
  cdef double [::1] eta01_view = eta01
  cdef double [::1] eta10_view = eta10
  cdef int i, j
  for i in range(m - 1):
    j = i + 1
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
