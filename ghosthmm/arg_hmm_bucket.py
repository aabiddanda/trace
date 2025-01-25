from copy import deepcopy

import numpy as np
import tskit
from joblib import Parallel, delayed
from KDEpy import FFTKDE
from pomegranate.distributions import *
from pomegranate.gmm import GeneralMixtureModel
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import digamma
from scipy.special import logsumexp as logsumexp_sp
from scipy.special import polygamma
from scipy.stats import expon as expon_sp
from scipy.stats import gamma as gamma_sp
from scipy.stats import norm as norm_sp
from scipy.stats import poisson as poisson_sp
from tqdm import tqdm

from .arg_hmm_utils import (
    backward_algo,
    backward_algo_product,
    ecm_full_update,
    emission_coal_event,
    emission_product_z1,
    emission_product_z1_flat,
    exp_coal_event,
    forward_algo,
    forward_algo_product,
    marginal_loglik,
    no_coal_event,
    posterior_assignment,
)
from .utils import Performance_utils


class DistinguishedLineages:
    """Class to estimate the number of coalescent events on distinguished lineages."""

    def __init__(self):
        from scipy.linalg import expm

    def create_Q(self, n):
        """Create coalescent rate-matrix."""
        assert n > 1
        return create_Q_cython(n)

    def prob_focal_coal_t(self, n=10, nt0=8, nt1=4, t_admix=0.05, t_interval=0.05):
        """
        Calculate the conditional probability on t1,t2.

        Args:
            n - haploid sample size
            nt0 - number of lineages at time t_admix
            nt1 - number of lineages at time t_admix + t_interval
            t_admix - starting point of lineage tracking
            t_interval - interval extent for tracking number of lineages
        Return:
            probability of "distinguished" coalescent events conditional on nt0 and nt1
        """
        assert n > 1
        assert t_admix > 0
        assert t_interval > 0
        Q = self.create_Q(n=n)
        P1 = expm(t_admix * Q)
        P2 = expm(t_interval * Q)
        ps = np.array([2.0 / j for j in np.arange(nt0, nt1, -1)])
        return np.sum(ps) * P1[0, n - nt0] * P2[n - nt0, n - nt1]

    def exp_focal_coal_interval(self, n=10, t_interval=0.1):
        """Compute the expected number of coalescent events in between the
        interval of [t_admix, t_admix + t_interval).

        Args:
            n - haploid sample size.
            t_interval - size of the window in coalescent units.

        Return:
            expected number of "distinguisehd coalescecens"
        """
        assert n > 1
        assert t_interval > 0
        tot_expectation = exp_focal_coal_interval_cython(n=n, t=t_interval)
        return tot_expectation


class GhostGammaHMM:
    """General implementation of ARG HMM for either the TMRCA or branch-bound based methods."""

    def __init__(
        self,
        t_admix=1999,
        admix_prop=0.03,
        alpha_upper_intro=100,
        beta_upper_intro=100,
        alpha_upper_normal=100,
        beta_upper_normal=100,
        chromosome=1,
    ):
        """Initialize the ARG-HMM-BRANCH class."""
        assert (admix_prop > 0) and (admix_prop < 1.0)
        self.t_admix = t_admix
        self.admix_prop = admix_prop
        self.alpha_upper_intro = alpha_upper_intro
        self.beta_upper_intro = beta_upper_intro
        self.alpha_upper_normal = alpha_upper_normal
        self.beta_upper_normal = beta_upper_normal
        self.pos = None
        self.ts = None
        self.t = None
        self.t_bulk = None
        self.t_lower = None
        self.t_upper = None
        self.bulk_t_lower = None
        self.bulk_t_upper = None
        self.emission_bins = None
        self.emissions = None
        self.p = 0.01
        self.q = 0.1
        self.chrom = chromosome
        self.intro_node = None
        self.bulk_intro_node = None

    def __repr__(self):
        """Representation of HMM in text."""
        rep_str = f"t_admix: {self.t_admix}\n"
        rep_str += f"admix_prop: {self.admix_prop}\n"
        rep_str += f"alpha_upper_intro: {self.alpha_upper_intro}\n"
        rep_str += f"beta_upper_intro: {self.beta_upper_intro}\n"
        rep_str += f"alpha_upper_normal: {self.alpha_upper_normal}\n"
        rep_str += f"beta_upper_normal: {self.beta_upper_normal}\n"
        rep_str += f"normal -> archaic: {self.p}\n"
        rep_str += f"archaic -> normal: {self.q}\n"
        return rep_str

    def add_tree_sequence(self, ts):
        """Add in a tree-sequence for analysis.

        The positions extracted are the midpoints of trees.
        """
        self.ts = ts
        assert ts.num_trees > 1
        self.m = ts.num_trees
        self.treespan = dict({})
        self.pos = np.zeros(self.m)
        for i, t in enumerate(ts.trees()):
            self.pos[i] = (t.interval.right + t.interval.left) / 2.0
            self.treespan[i] = [t.interval.left, t.interval.right]

    def extract_tmrca(self, i, js, f=np.mean):
        """Extract TMRCA as the primary feature for downstream modeling.

        The extracted TMRCAs can be modeled using a transformation function as well.
        """
        assert self.ts is not None
        assert i < self.ts.num_samples
        assert js.size > 0
        self.t = np.array(
            [f([tree.tmrca(i, j) for j in js if i != j]) for tree in self.ts.trees()]
        )
        assert self.t.size == self.m
        self.t_admix = 0.0

    def extract_tmrca_bulk(self, population="AFR", nsamp=5, f=np.mean):
        """Extract the TMRCA between a set of samples at each locus.

        NOTE: this does not set the focal estimate of `t` that we need for actual inference.
        """
        assert self.ts is not None
        self.t_bulk = np.zeros(shape=(nsamp, self.m))
        js = [i for i in self.ts.samples(population=population)]
        for p, tree in enumerate(self.ts.trees()):
            for i, idx in enumerate(js[:nsamp]):
                self.t_bulk[i, p] = f([tree.tmrca(idx, j) for j in js if idx != j])
        self.t_admix = 0.0

    def extract_branch_boundaries(self, i):
        """Extract the branch length subtending sample i."""
        assert self.ts is not None
        assert i < self.ts.num_samples
        intro_node = []
        upper_branch = []
        for tree in self.ts.trees():
            lower = []
            lid = []
            u = i
            while u != tskit.NULL:
                lower.append(tree.time(u))
                lid.append(u)
                u = tree.parent(u)
            if u == tskit.NULL:
                lower.append(tree.time(tree.roots[0]))
                lid.append(u)
            lower = np.array(lower)
            lid = np.array(lid)
            lower_node = lid[np.argwhere(lower <= self.t_admix)[-1]]
            upper_node = lower[np.argwhere(lower >= self.t_admix)[0]]
            intro_node.append(lower_node)
            upper_branch.append(upper_node)
        self.intro_node = np.array(intro_node)
        self.t_upper = np.array(upper_branch)
        self.t = np.array(upper_branch)
        return upper_branch, intro_node

    def extract_branch_boundaries_bulk(self, population="AFR", nsamp=5):
        """Extract the branch length subtending many samples.

        Note: this does not set the focal estimate of t for actual inference.
        """
        assert self.ts is not None
        self.bulk_t_upper = np.zeros(shape=(nsamp, self.m))
        self.bulk_intro_node = np.zeros(shape=(nsamp, self.m))
        samples = [i for i in self.ts.samples(population=population)]
        for j, i in enumerate(samples[:nsamp]):
            (
                self.bulk_t_upper[j],
                self.bulk_intro_node[j],
            ) = self.extract_branch_boundaries(i)
        assert np.all(self.bulk_t_upper >= self.t_admix)

    def extract_crosspop_branch_boundaries(self, i):
        """Extract the upper branch as the first branch subtending a leaf in a different population."""
        assert self.ts is not None
        assert i < self.ts.num_samples
        assert self.ts.num_populations > 1
        intro_node = []
        upper_branch = []
        for tree in self.ts.trees():
            lower = [0]
            lid = [i]
            u = i
            while u != tskit.NULL and (np.max(lower) < self.t_admix):
                lower.append(tree.time(u))
                lid.append(u)
                pops = np.array([tree.population(s) for s in tree.samples(u)])
                if ~np.all(pops != pops[0]) and (tree.time(u) > self.t_admix):
                    u = tskit.NULL
                else:
                    u = tree.parent(u)
            lower = np.array(lower)
            lid = np.array(lid)
            lower_node = lid[np.argwhere(lower <= self.t_admix)[-1]]
            upper_node = lower[-1]
            intro_node.append(lower_node[0])
            upper_branch.append(upper_node)
        self.intro_node = np.array(intro_node)
        self.t_upper = np.array(upper_branch)
        self.t = np.array(upper_branch)
        return upper_branch, intro_node

    def extract_crosspop_branch_boundaries_bulk(
        self, population="AFR", nsamp=5, seed=42
    ):
        """Extract the branch length subtending many samples."""
        np.random.seed(seed)
        assert self.ts is not None
        samples = [i for i in self.ts.samples(population=population)]
        self.bulk_t_upper = np.zeros(shape=(nsamp, self.m))
        self.bulk_intro_node = np.zeros(shape=(nsamp, self.m))
        for j, i in enumerate(samples[:nsamp]):
            (
                self.bulk_t_upper[j],
                self.bulk_intro_node[j],
            ) = self.extract_crosspop_branch_boundaries(i)
        assert np.all(self.bulk_t_upper >= self.t_admix)

    def add_recombination_map(self, recmap):
        """Add and interpolate a recombination rate between every location.

        Recmap must be a two column numpy array (with position and Morgans)
        """
        assert self.ts is not None
        assert recmap.ndim == 2
        assert np.all(np.diff(recmap[:, 0]) > 0)
        assert np.all(np.diff(recmap[:, 1]) > 0)
        interp_recmap = interp1d(recmap[:, 0], recmap[:, 1])
        self.recmap = interp_recmap(self.pos)

    def set_constant_recomb(self, rate=1e-8):
        """Set a constant recombination rate per basepair per generation."""
        assert self.ts is not None
        assert self.pos.size > 1
        assert (rate > 0.0) and (rate < 1.0)
        self.recmap = rate * self.pos

    def set_emission_binning(self, bin_edges=None, n="auto", mode="focal"):
        """Set the binning of the emission to be a discrete distribution."""
        assert isinstance(n, int) or (n == "auto")
        if isinstance(n, int):
            assert n > 1
        if bin_edges is None:
            if mode == "bulk":
                _, bin_edges = np.histogram(
                    self.bulk_t_upper,
                    n,
                    range=(self.t_admix, 1e6),
                )
            elif mode == "focal":
                _, bin_edges = np.histogram(self.t, n, range=(self.t_admix, 1e6))
            else:
                raise ValueError("mode argument must be either `bulk` or `focal`")
        assert np.min(bin_edges) <= self.t_admix
        self.emission_bins = bin_edges

    def set_emission_dist(self, eps=1e-8):
        """Set the emission distribution with a pseudo-count."""
        assert self.emission_bins is not None
        n = self.emission_bins.size
        emissions = np.zeros(shape=(2, n))
        emissions[0, :] = gamma_sp.cdf(
            self.emission_bins,
            self.alpha_upper_normal,
            scale=1.0 / self.beta_upper_normal,
            loc=self.t_admix,
        )
        emissions[1, :] = gamma_sp.cdf(
            self.emission_bins,
            self.alpha_upper_intro,
            scale=1.0 / self.beta_upper_intro,
            loc=self.t_admix,
        )
        emissions[0, 1:] = emissions[0, 1:] - emissions[0, :-1]
        emissions[1, 1:] = emissions[1, 1:] - emissions[1, :-1]
        emissions += eps
        emissions[0, :] /= np.sum(emissions[0, :])
        emissions[1, :] /= np.sum(emissions[1, :])
        self.emissions = emissions

    def emission_prob(self, t_upper, x):
        """Compute the emission probability for the tmrca."""
        if x == 0:
            return gamma_sp.pdf()
        else:
            return gamma_sp.pdf()

    def forward_algorithm(self):
        """Run a forward pass through the HMM.

        Return the forward probability and the logliklihood.
        """
        alphas = np.zeros(shape=(2, self.m))
        # Setting up some initialization here ...
        alphas[0, 0] = np.log(1.0 - self.admix_prop) + self.emission_prob(
            self.t[0], x=0
        )
        alphas[1, 0] = np.log(self.admix_prop) + self.emission_prob(self.t[0], x=1)
        r_pi, r_qi = self.p, self.q
        for i in range(1, self.m):
            alphas[0, i] = self.emission_prob(self.t[i], x=0) + np.logaddexp(
                np.log(1.0 - r_pi) + alphas[0, (i - 1)],
                np.log(r_qi) + alphas[1, (i - 1)],
            )
            alphas[1, i] = self.emission_prob(self.t[i], x=1) + np.logaddexp(
                np.log(r_pi) + alphas[0, i - 1], np.log(1.0 - r_qi) + alphas[1, i - 1]
            )
        loglik = logsumexp(alphas[:, -1])
        return alphas, loglik

    def backward_algorithm(self):
        """Run a backward pass through the HMM for focal TMRCAs."""
        betas = np.zeros(shape=(2, self.m))
        betas[:, -1] = np.log([1.0, 1.0])
        r_pi, r_qi = self.p, self.q
        for i in range(self.m - 2, -1, -1):
            betas[0, i] = np.logaddexp(
                np.log(r_pi) + self.emission_prob(self.t[i + 1], x=1) + betas[1, i + 1],
                np.log(1.0 - r_pi)
                + self.emission_prob(self.t[i + 1], x=0)
                + betas[0, i + 1],
            )
            betas[1, i] = np.logaddexp(
                np.log(r_qi) + self.emission_prob(self.t[i + 1], x=0) + betas[0, i + 1],
                np.log(1.0 - r_qi)
                + self.emission_prob(self.t[i + 1], x=1)
                + betas[1, i + 1],
            )
        # reconfiguring to use the initial distribution here...
        betas[0, 0] = (
            np.log(1.0 - self.admix_prop)
            + self.emission_prob(self.t[0], x=0)
            + betas[0, 0]
        )
        betas[1, 0] = (
            np.log(self.admix_prop) + self.emission_prob(self.t[0], x=1) + betas[1, 0]
        )
        loglik = logsumexp(betas[:, 0])
        return betas, loglik

    def composite_likelihood(self, mode="branch"):
        """Evaluate the composite likelihood of tmrcas or branch lengths."""
        if mode not in ["tmrca", "branch"]:
            raise ValueError(
                f"Mode should be either `tmrca|branch!` but is instead {mode}"
            )
        if mode == "tmrca":
            txs = self.t_bulk
        else:
            txs = self.bulk_t_upper
        alphas = []
        for j in range(txs.shape[0]):
            self.t = txs[j, :]
            cur_alpha = self.forward_algorithm()
            alphas.append(cur_alpha)
        alphas = np.array(alphas)
        # We mainly keep the previous parameters for this here...
        total_loglik = logsumexp(alphas[:, :, -1])
        return total_loglik

    def forward_backward(self):
        """Compute the posterior decoding of archaic admixture across trees."""
        alphas, _ = self.forward_algorithm()
        betas, _ = self.backward_algorithm()
        # assure that the likelihoods are close to one another ...
        try:
            assert np.isclose(logsumexp(alphas[:, -1]), logsumexp(betas[:, 0]))
        except AssertionError:
            print(logsumexp(alphas[:, -1]), logsumexp(betas[:, 0]))
        gammas = alphas + betas
        gammas_sum = np.zeros(shape=(1, self.m))
        for i in range(0, self.m):
            gammas_sum[0, i] = np.logaddexp(gammas[0, i], gammas[1, i])
        gammas_normed = gammas - gammas_sum
        return gammas_normed, alphas, betas

    def est_gamma_mle(self, xs, niter=10):
        """Perform estimation of MLE parameters for a gamma distribution.

        The initialization and method are based on: https://tminka.github.io/papers/minka-gamma.pdf
        """
        assert np.all(xs > 0.0)
        mean_log = np.mean(np.log(xs))
        log_mean = np.log(np.mean(xs))
        cur_a = 0.5 / (log_mean - mean_log)
        for _ in range(niter):
            a_est = mean_log - log_mean + np.log(cur_a) - digamma(cur_a)
            a_est /= (cur_a**2) * (1 / cur_a - polygamma(1, cur_a))
            a_est += 1 / cur_a
            cur_a = 1.0 / a_est
        b_est = 1.0 / (np.mean(xs) / cur_a)
        return (cur_a, b_est)

    def ecm_update1(self, xs, zs0, zs1, b0, b1):
        """Expect conditional maximization for shape parameter."""
        from scipy.optimize import brentq
        from scipy.special import digamma

        clear_sum = np.sum(np.exp(zs0) * (np.log(xs) + np.log(b0))) + np.sum(
            np.exp(zs1) * (np.log(xs) + np.log(b1))
        )
        f = lambda a: clear_sum - xs.size * digamma(a)  # noqa
        a_hat = brentq(f, 0.0, 1e6)
        return a_hat

    def ecm_update2(self, xs, zs, a):
        """Expect conditional maximization for scale parameters."""
        return a * np.exp(logsumexp(zs)) / np.sum(xs * np.exp(zs))

    def estimate_params_update(self, txs, alphas, betas, gammas):
        """Estimating parameters via an EM-Update Step.

        txs: repeated estimates of tmrca or upper branch (`np.array`)
        alphas: repeated estimates of the forward algorithm (`np.array`)
        betas: repeated estimates of the backward algorithm (`np.array`)
        gammas: repeated estimates of the forward-backward normed (`np.array`)

        This should then be followed up by placing these estimates into the class and calling:
        self.set_emission_dist()

        which will update the emission distribution.
        The forward_backward algorithm should be run on each individual for this instance.
        """
        assert txs.shape[0] == alphas.shape[0]
        assert txs.shape[0] == betas.shape[0]
        assert alphas.shape == betas.shape
        assert betas.shape == gammas.shape

        alpha_shared = self.ecm_update1(
            xs=txs - self.t_admix,
            zs0=gammas[:, 0, :],
            zs1=gammas[:, 1, :],
            b0=self.beta_upper_normal,
            b1=self.beta_upper_intro,
        )
        alpha_null = alpha_shared
        alpha_intro = alpha_shared
        beta_null = self.ecm_update2(
            xs=txs - self.t_admix, zs=gammas[:, 0, :], a=alpha_shared
        )
        beta_intro = self.ecm_update2(
            xs=txs - self.t_admix, zs=gammas[:, 1, :], a=alpha_shared
        )
        if beta_intro >= beta_null:
            # flip them so that we can interpret this more easily...
            tmp = beta_null
            beta_null = beta_intro
            beta_intro = tmp

        # now we have to estimate the transition probabilities ...
        zeta_01 = np.zeros(txs.shape)
        zeta_10 = np.zeros(txs.shape)
        norm_factor = logsumexp(alphas[:, :, -1], axis=1)
        for i, j in zip(np.arange(txs.shape[1] - 1), np.arange(1, txs.shape[1])):
            emissions0 = np.array([self.emission_prob(t, 0) for t in txs[:, j]])
            emissions1 = np.array([self.emission_prob(t, 1) for t in txs[:, j]])
            zeta_01[:, i] = (
                alphas[:, 0, i]
                + np.log(self.p)
                + emissions1
                + betas[:, 1, j]
                - norm_factor
            )
            zeta_10[:, i] = (
                alphas[:, 1, i]
                + np.log(self.q)
                + emissions0
                + betas[:, 0, j]
                - norm_factor
            )
        p_est = np.exp(logsumexp(zeta_01[:, :-1]) - logsumexp(gammas[:, 0, :-1]))
        q_est = np.exp(logsumexp(zeta_10[:, :-1]) - logsumexp(gammas[:, 1, :-1]))
        w_est = np.mean(np.exp(gammas[:, 1, 0]))
        # w_est = np.clip(w_est, 1e-4, 1-1e-4)
        if w_est >= 0.5:
            # We need to flip the categories (similar to betas above)
            w_est = 1.0 - w_est
        return (
            alpha_null,
            beta_null,
            alpha_intro,
            beta_intro,
            w_est,
            p_est,
            q_est,
        )

    def composite_baum_welch(self, mode="tmrca", niter=10):
        """Estimation of parameters via the Baum-Welch Algorithm on the composite Likelihood."""
        if mode not in ["tmrca", "branch"]:
            raise ValueError(
                f"Mode should be either `tmrca|branch!` but is instead {mode}"
            )
        if mode == "tmrca":
            txs = self.t_bulk
        else:
            txs = self.bulk_t_upper
        assert niter >= 0
        self.t = txs[0, :]
        self.set_emission_binning()
        self.set_emission_dist()
        loglik_acc = np.zeros(niter)
        alpha_null_acc = np.zeros(niter)
        beta_null_acc = np.zeros(niter)
        alpha_intro_acc = np.zeros(niter)
        beta_intro_acc = np.zeros(niter)
        p_est_acc = np.zeros(niter)
        q_est_acc = np.zeros(niter)
        w_est_acc = np.zeros(niter)
        for i in tqdm(range(niter)):
            alphas = []
            betas = []
            gammas = []
            # Learning all of the parameters
            # NOTE: this is where we could parallelize more ...
            for j in range(txs.shape[0]):
                self.t = txs[j, :]
                cur_gamma, cur_alpha, cur_beta = self.forward_backward()
                gammas.append(cur_gamma)
                alphas.append(cur_alpha)
                betas.append(cur_beta)
            alphas = np.array(alphas)
            betas = np.array(betas)
            gammas = np.array(gammas)
            # We mainly keep the previous parameters for this here...
            total_loglik = logsumexp(alphas[:, :, -1])
            loglik_acc[i] = total_loglik
            alpha_null_acc[i] = self.alpha_upper_normal
            beta_null_acc[i] = self.beta_upper_normal
            alpha_intro_acc[i] = self.alpha_upper_intro
            beta_intro_acc[i] = self.beta_upper_intro
            p_est_acc[i] = self.p
            q_est_acc[i] = self.q
            w_est_acc[i] = self.admix_prop
            (
                alpha_null,
                beta_null,
                alpha_intro,
                beta_intro,
                w_est,
                p_est,
                q_est,
            ) = self.estimate_params_update(txs, alphas, betas, gammas)
            self.alpha_upper_normal = alpha_null
            self.beta_upper_normal = beta_null
            self.alpha_upper_intro = alpha_intro
            self.beta_upper_intro = beta_intro
            self.admix_prop = w_est
            self.p = p_est
            self.q = q_est
            self.set_emission_binning()
            self.set_emission_dist()
        param_dict = {
            "iteration": np.arange(niter),
            "composite_loglik": loglik_acc,
            "alpha_null": alpha_null_acc,
            "beta_null": beta_null_acc,
            "alpha_intro": alpha_intro_acc,
            "beta_intro": beta_intro_acc,
            "p": p_est_acc,
            "q": q_est_acc,
            "w": w_est_acc,
        }
        return param_dict

    def simulate_data(self, seed=42, p=0.1, q=0.1):
        """Simulate branch-length data from the HMM."""
        np.random.seed(seed)
        zs = np.zeros(self.m)
        zs[0] = np.random.choice([0, 1], p=[1 - self.admix_prop, self.admix_prop])
        for i in range(1, self.m):
            if zs[i - 1]:
                zs[i] = np.random.choice([1, 0], p=[1 - q, q])
            else:
                zs[i] = np.random.choice([1, 0], p=[p, 1 - p])
        xs = np.zeros(self.m)
        n_archaic = np.sum(zs == 1)
        xs[zs == 1] = gamma_sp.rvs(
            self.alpha_upper_intro,
            loc=self.t_admix,
            scale=1.0 / self.beta_upper_intro,
            size=n_archaic,
        )
        xs[zs == 0] = gamma_sp.rvs(
            self.alpha_upper_normal,
            loc=self.t_admix,
            scale=1.0 / self.beta_upper_normal,
            size=(self.m - n_archaic),
        )
        return xs, zs


class GhostFixedHmm:
    def __init__(self):
        """Initialization of the class."""
        self.ts = None
        self.xss = None
        self.nt1 = None
        self.nt2 = None
        self.lower = None
        self.upper = None
        self.exp_coal = None
        self.pi0_lower = 0.5
        self.pos = None
        self.treespan = None

    def add_tree_sequence(self, ts):
        """Add in a tree-sequence for analysis.

        The positions extracted are the midpoints of trees.
        """
        self.ts = ts
        assert ts.num_trees > 1
        self.n_trees = ts.num_trees
        self.treespan = dict({})
        self.pos = np.zeros(self.n_trees)
        for i, t in enumerate(ts.trees()):
            self.pos[i] = np.mean(t.interval)
            self.treespan[i] = np.array([t.interval.left, t.interval.right])
            assert self.treespan[i][1] >= self.treespan[i][0]

    def get_cM(self, pos, map_dict):
        """Helper function to get genetic position given a genetic map and physical position."""
        map_pos = sorted(map_dict.keys())
        out = 0
        if pos < map_pos[0]:
            out = 0
        elif pos >= map_pos[-1]:
            out = map_dict[map_pos[-1]][2]
        else:
            for p in range(len(map_pos)):
                if pos >= map_pos[p] and pos < map_pos[p + 1]:
                    out = (
                        map_dict[map_pos[p]][2]
                        + (pos - map_pos[p]) * map_dict[map_pos[p]][1] * 1e-6
                    )
                    break
        return out

    def read_recombination_map(self, map=None):
        """Read in a plink format recombination map."""
        if map is None:
            # Just assume 1cM ~ 1 MB ...
            for i in self.treespan.keys():
                self.treespan[i] = self.treespan[i] / 1e6
                self.pos[i] = np.mean(self.treespan[i])
        else:
            infile = open(map)
            lines = infile.readlines()
            infile.close()
            rec = dict({})
            for i in range(len(lines)):
                s = lines[i].strip("\n").strip("\t").split()
                rec[int(s[1])] = [float(s[2]), float(s[3])]
            for i in self.treespan.keys():
                self.treespan[i][0] = self.get_cM(self.treespan[i][0], rec)
                self.treespan[i][1] = self.get_cM(self.treespan[i][1], rec)
                assert self.treespan[i][1] >= self.treespan[i][0]
                # Set the position ...
                self.pos[i] = np.mean(self.treespan[i])

    def expected_n_coal(self):
        """Expected number of coalescent events in this interval.

        NOTE: t2 > t1, so we expect more lineages to be in t1's range ...
        """
        assert self.nt1 is not None
        assert self.nt2 is not None
        assert self.nt1.size == self.nt2.size
        exp_coal = np.array(
            [exp_coal_event(n1, n2) for (n1, n2) in zip(self.nt1, self.nt2)]
        )
        no_coal = np.array(
            [no_coal_event(n1, n2) for (n1, n2) in zip(self.nt1, self.nt2)]
        )
        self.exp_coal = exp_coal
        self.mean_coal = np.mean(exp_coal)
        self.mean_no_coal = np.mean(no_coal)

    def extract_coal_events(self, idx=[0], t1=100, t2=1000):
        """Method to extract the number of coalescent events features for the HMM.
        Arguments:

        - ts: tskit TreeSequence containing inferred ARG
        - idx: haplotype index of test haplotypes (list)
        - t1: time in generations for lower bound (tadmix)
        - t2: time in generations for upper bound (tarchaic)

        """
        assert t2 >= t1
        assert self.ts is not None
        assert self.ts.num_samples > 0
        # Assure that the number of coalescent events is accounted for apriori
        n = self.ts.num_trees
        xss = np.zeros(shape=(len(idx), n))
        lower = np.zeros(shape=(len(idx), n))
        upper = np.zeros(shape=(len(idx), n))
        root_t = np.zeros(n, dtype=np.float32)
        ts1 = self.ts.decapitate(t1)
        ts2 = self.ts.decapitate(t2)
        self.nt1 = np.array([tx.num_roots for (_, _, tx) in self.ts.coiterate(ts1)])
        self.nt2 = np.array([tx.num_roots for (_, _, tx) in self.ts.coiterate(ts2)])
        pos = np.array([np.mean(tree.interval) for tree in self.ts.trees()])
        for j, x in enumerate(idx):
            xs = np.zeros(n, dtype=np.int32)
            lower_time = np.zeros(n, dtype=np.int32)
            upper_time = np.zeros(n, dtype=np.int32)
            for i, tree in enumerate(self.ts.trees()):
                tx = []
                k = x
                while k != tskit.NULL:
                    tx.append(tree.time(k))
                    k = tree.parent(k)
                tx = np.array(tx)
                lower_time[i] = tx[np.argwhere(tx < t2)[-1]]
                if tree.time(tree.root) >= t2:
                    upper_time[i] = tx[np.argwhere(tx > t2)[0]]
                else:
                    upper_time[i] = tree.time(tree.root)
                xs[i] = np.sum((tx <= t2) & (tx >= t1))
                # root_t[i] = tree.time(tree.root)
            xss[j] = xs
            lower[j] = lower_time
            upper[j] = upper_time
        self.xss = xss
        self.lower = lower
        self.upper = upper
        self.expected_n_coal()
        return xss, self.nt1, self.nt2, pos

    def loglik(self, xs, p=0.01, q=0.05, pi0=0.5, lamb0=1.0):
        """Log-likelihood function"""
        loglik = marginal_loglik(
            xs=xs,
            p=p,
            q=q,
            pi0=pi0,
            lamb_null=self.exp_coal,
            lamb0=lamb0,
            gen_pos=self.pos,
        )
        return loglik

    def composite_loglik(self, p=0.01, q=0.05, pi0=0.5, lamb0=1.0, n_jobs=2):
        """Composite log-likelihood function across all estimates."""
        assert self.xss is not None
        comp_loglik = 0.0
        logliks = Parallel(n_jobs=n_jobs)(
            delayed(marginal_loglik)(
                xs=self.xss[i, :],
                p=p,
                q=q,
                pi0=pi0,
                lamb_null=self.exp_coal,
                lamb0=lamb0,
                gen_pos=self.pos,
            )
            for i in range(self.xss.shape[0])
        )
        comp_loglik = sum(logliks)
        return comp_loglik

    def estimate_transition_probs(self, **kwargs):
        """Estimate the transition probabilities using numerical optimization.

        NOTE: these are optimizing the actual `rate` parameters for each transition mode.
        """
        f = lambda a: -self.composite_loglik(p=a[0], q=a[1], **kwargs)  # noqa
        opt_res = minimize(
            f,
            x0=[1, 1],
            bounds=[(0.1, 50.0), (0.1, 50.0)],
            tol=1e-3,
            method="Powell",
            options={"disp": True},
        )
        p_hat = opt_res.x[0]
        q_hat = opt_res.x[1]
        return p_hat, q_hat

    def set_pi0_lower(self, lower):
        """Set specific lower bound for pi0."""
        self.pi0_lower = lower

    def estimate_full_parameters(self, **kwargs):
        """Estimate the full set of parameters."""
        f = lambda a: -self.composite_loglik(
            p=a[0], q=a[1], pi0=a[2], lamb0=a[3], **kwargs
        )  # noqa
        opt_res = minimize(
            f,
            x0=[1.0, 1.0, (self.pi0_lower + 0.9999) / 2, self.mean_coal / 2],
            bounds=[
                (0.1, 50.0),
                (0.1, 50.0),
                (self.pi0_lower, 0.9999),
                (1e-4, self.mean_coal),
            ],
            tol=1e-3,
            method="Powell",
            options={"disp": True},
        )
        p_hat = opt_res.x[0]
        q_hat = opt_res.x[1]
        pi0_hat = opt_res.x[2]
        lamb0_hat = opt_res.x[3]
        return p_hat, q_hat, pi0_hat, lamb0_hat

    def forward_algorithm(self, xs, p=1e-2, q=1e-2, pi0=0.5, lamb0=1.0):
        """Implement the forward algorithm for the binary hmm."""
        assert xs.ndim == 1
        # assert (p > 0) and (q > 0)
        # assert (p < 1) and (q < 1)
        assert lamb0 >= 0
        assert (pi0 >= 0) and (pi0 <= 1.0)
        assert xs.size == self.exp_coal.size
        assert self.pos.size == self.exp_coal.size
        alphas, scaler, loglik = forward_algo(
            xs,
            p=p,
            q=q,
            pi0=pi0,
            lamb_null=self.exp_coal,
            lamb0=lamb0,
            gen_pos=self.pos,
        )
        return alphas, scaler, loglik

    def backward_algorithm(self, xs, p=1e-2, q=1e-2, pi0=0.5, lamb0=1.0):
        """The backward algorithm for the ghost admixture HMM model."""
        assert xs.ndim == 1
        # assert (p > 0) and (q > 0)
        # assert (p < 1) and (q < 1)
        assert lamb0 >= 0
        assert (pi0 >= 0) and (pi0 <= 1.0)
        assert self.pos.size == self.exp_coal.size
        betas, scaler, loglik = backward_algo(
            xs,
            p=p,
            q=q,
            pi0=pi0,
            lamb_null=self.exp_coal,
            lamb0=lamb0,
            gen_pos=self.pos,
        )
        return betas, scaler, loglik

    def forward_backward(self, xs, p=1e-2, q=1e-2, pi0=0.5, lamb0=1.0):
        """Computing the forward-backward algorithm.

        Arguments:
          - xs: number of coalescent events affecting focal lineage
          - pi0: mixture proportion for reduction in Poisson emission
          - p: transition probability from unadmixed -> admixed
          - q: transition probability from admixed -> unadmixed
          - ey_true: expected number of coalescent events in this time interval
          - lamb0: poisson rate parameter for zero inflated poisson
        """
        alphas, _, loglik = self.forward_algorithm(
            xs=xs, p=p, q=q, pi0=pi0, lamb0=lamb0
        )
        betas, _, _ = self.backward_algorithm(xs=xs, p=p, q=q, pi0=pi0, lamb0=lamb0)
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas

    def eval_Jhat(
        self, nboots=5, p=1e-2, q=1e-2, pi0=0.5, lamb0=1.0, eps=1e-3, seed=42, **kwargs
    ):
        """Estimate the variability matrix across all parameters holding the others constant."""
        assert (eps > 0.0) and (eps <= 1e-1)
        assert self.xss is not None
        assert self.xss.shape[0] > 0
        assert seed > 0
        mat_U_gradients = np.zeros(shape=(nboots, 4, 4))
        orig_xss = self.xss.copy()
        n = self.xss.shape[0]
        for i in range(nboots):
            cur_U = np.zeros(shape=(4, 4))
            idx = np.random.choice(np.arange(n), size=n)
            self.xss = self.xss[idx, :]
            # 1. Gradient for p function (second derivative)
            xs = (p - eps * p, p, p + eps * p)
            ys = [
                self.composite_loglik(p=xs[0], q=q, pi0=pi0, lamb0=lamb0),
                self.composite_loglik(p=xs[1], q=q, pi0=pi0, lamb0=lamb0),
                self.composite_loglik(p=xs[2], q=q, pi0=pi0, lamb0=lamb0),
            ]
            spl = UnivariateSpline(xs, ys, **kwargs)
            cur_U[0, 0] = spl.derivatives(p)[1]
            # 2. Gradient for q function (second derivative)
            xs = (q - eps * q, q, q + eps * q)
            ys = [
                self.composite_loglik(p=p, q=xs[0], pi0=pi0, lamb0=lamb0),
                self.composite_loglik(p=p, q=xs[1], pi0=pi0, lamb0=lamb0),
                self.composite_loglik(p=p, q=xs[2], pi0=pi0, lamb0=lamb0),
            ]
            spl = UnivariateSpline(xs, ys, **kwargs)
            cur_U[1, 1] = spl.derivatives(q)[1]
            # 3. Gradient for pi0 function (second derivative)
            xs = (pi0 - eps * pi0, pi0, pi0 + eps * pi0)
            ys = [
                self.composite_loglik(p=p, q=q, pi0=xs[0], lamb0=lamb0),
                self.composite_loglik(p=p, q=q, pi0=xs[1], lamb0=lamb0),
                self.composite_loglik(p=p, q=q, pi0=xs[2], lamb0=lamb0),
            ]
            spl = UnivariateSpline(xs, ys, **kwargs)
            cur_U[2, 2] = spl.derivatives(pi0)[1]
            # 4. Gradient for lamb0 function (second derivative)
            xs = (lamb0 - eps * lamb0, lamb0, lamb0 + eps * lamb0)
            ys = [
                self.composite_loglik(p=p, q=q, pi0=pi0, lamb0=xs[0]),
                self.composite_loglik(p=p, q=q, pi0=pi0, lamb0=xs[1]),
                self.composite_loglik(p=p, q=q, pi0=pi0, lamb0=xs[2]),
            ]
            spl = UnivariateSpline(xs, ys, **kwargs)
            cur_U[3, 3] = spl.derivatives(lamb0)[1]
            mat_U_gradients[i, :, :] = cur_U
        self.xss = orig_xss
        J_hat = np.zeros(shape=(4, 4))
        for i in range(nboots):
            J_hat += mat_U_gradients[i, :, :] @ mat_U_gradients[i, :, :].T
        J_hat = J_hat / nboots
        return J_hat

    def eval_hessian(
        self, p=1e-2, q=1e-2, pi0=0.5, lamb0=1.0, eps=1e-3, seed=42, **kwargs
    ):
        """Evaluate the Hessian matrix centered on the composite MLE parameters."""
        assert (eps > 0.0) and (eps <= 1e-1)
        assert self.xss is not None
        assert self.xss.shape[0] > 0
        assert seed > 0
        H_hat = np.zeros(shape=(4, 4))
        # 1. Gradient for p function (second derivative)
        xs = (p - eps * p, p, p + eps * p)
        ys = [
            self.composite_loglik(p=xs[0], q=q, pi0=pi0, lamb0=lamb0),
            self.composite_loglik(p=xs[1], q=q, pi0=pi0, lamb0=lamb0),
            self.composite_loglik(p=xs[2], q=q, pi0=pi0, lamb0=lamb0),
        ]
        spl = UnivariateSpline(xs, ys, **kwargs)
        H_hat[0, 0] = spl.derivatives(p)[2]
        # 2. Gradient for q function (second derivative)
        xs = (q - eps * q, q, q + eps * q)
        ys = [
            self.composite_loglik(p=p, q=xs[0], pi0=pi0, lamb0=lamb0),
            self.composite_loglik(p=p, q=xs[1], pi0=pi0, lamb0=lamb0),
            self.composite_loglik(p=p, q=xs[2], pi0=pi0, lamb0=lamb0),
        ]
        spl = UnivariateSpline(xs, ys, **kwargs)
        H_hat[1, 1] = spl.derivatives(q)[2]
        # 3. Gradient for pi0 function (second derivative)
        xs = (pi0 - eps * pi0, pi0, pi0 + eps * pi0)
        ys = [
            self.composite_loglik(p=p, q=q, pi0=xs[0], lamb0=lamb0),
            self.composite_loglik(p=p, q=q, pi0=xs[1], lamb0=lamb0),
            self.composite_loglik(p=p, q=q, pi0=xs[2], lamb0=lamb0),
        ]
        spl = UnivariateSpline(xs, ys, **kwargs)
        H_hat[2, 2] = spl.derivatives(pi0)[2]
        # 4. Gradient for lamb0 function (second derivative)
        xs = (lamb0 - eps * lamb0, lamb0, lamb0 + eps * lamb0)
        ys = [
            self.composite_loglik(p=p, q=q, pi0=pi0, lamb0=xs[0]),
            self.composite_loglik(p=p, q=q, pi0=pi0, lamb0=xs[1]),
            self.composite_loglik(p=p, q=q, pi0=pi0, lamb0=xs[2]),
        ]
        spl = UnivariateSpline(xs, ys, **kwargs)
        H_hat[3, 3] = spl.derivatives(lamb0)[2]
        return H_hat

    def eval_godambe(self, H_hat=None, J_hat=None, **kwargs):
        """Evaluate parameter uncertaintties using an approximation to the Godambe Information Matrix.

        Arguments:
            H_hat: 4x4 matrix of the estimated hessian
            J_hat: 4x4 matrix of the 'variability matrix' (Eq S3 from Coffman et al).

        Returns:
            G_hat: Estimated Godambe Information Matrix.
            param_SE: vector of standard-errors for each parameter

        """
        if H_hat is None:
            H_hat = self.eval_hessian(**kwargs)
        if J_hat is None:
            J_hat = self.eval_Jhat(**kwargs)
        assert H_hat.ndim == J_hat.ndim
        assert H_hat.shape[0] == J_hat.shape[0]
        assert H_hat.shape[1] == J_hat.shape[1]
        assert H_hat.shape[0] == H_hat.shape[0]
        # NOTE: we might have to use the pseudo-inverse here potentially.
        G_hat = H_hat @ np.linalg.inv(J_hat) @ H_hat
        param_SE = np.sqrt(np.diagonal(G_hat))
        return (G_hat, param_SE)


class GhostProductHmm:
    def __init__(self, t_admix=None, t_archaic=None):
        """Initialization of the class."""
        self.ts = None
        self.pos = None
        self.xss = None
        self.treespan = None
        self.f0 = None
        self.emissions = None
        self.emissions_split = None
        self.null_exp_coal = None
        self.intro_exp_coal = None
        self.alpha = 0.5
        self.a1 = None
        self.b1 = None
        self.a2 = None
        self.b2 = None
        self.pi0 = None
        self.p = None
        self.q = None
        self.t_admix = t_admix
        self.t_archaic = t_archaic
        self.padmix = None
        self.ne_archaic = None

    def update_transitions(self, gammas, alphas, betas, p, q, njobs):
        """Updates for the transitions."""
        assert alphas.size == betas.size
        assert (p > 0) and (q > 0)
        assert (p < 1) and (q < 1)
        m = self.m
        # if gammas.ndim > 2:
        #     inds = range(alphas.shape[0])
        #     eta_01 = np.zeros(shape=(len(inds), m - 1))
        #     eta_10 = np.zeros(shape=(len(inds), m - 1))
        # else:
        eta_01 = np.zeros(m - 1)
        eta_10 = np.zeros(m - 1)

        def update_oneind(alphas, betas, emissions, p, q):  # , ind = None):
            eta01, eta10 = update_oneind_cython(
                m=m,
                alphas=alphas,
                betas=betas,
                emissions=emissions,
                p=p,
                q=q,
            )
            # if ind is None:
            # else:
            #     alps = np.copy(alphas[ind], order="C")
            #     bets = np.copy(betas[ind], order="C")
            #     ems = np.copy(emissions[ind], order="C")
            #     eta01, eta10 = update_oneind_cython(
            #         m = m,
            #         alphas = alps,
            #         betas = bets,
            #         emissions = ems,
            #         p = p,
            #         q = q,
            #     )
            return eta01, eta10

        # if eta_01.ndim > 1:
        #     etas = Parallel(n_jobs=njobs)(delayed(update_oneind)(alphas, betas, self.emissions, p, q, ind) for ind in inds)
        #     for ind in range(len(etas)):
        #         eta_01[ind] = etas[ind][0]
        #         eta_10[ind] = etas[ind][1]
        #     p_est = np.exp(eta_01).sum() / np.exp(gammas[:, 0, :-1]).sum()
        #     q_est = np.exp(eta_10).sum() / np.exp(gammas[:, 1, :-1]).sum()
        # else:
        eta_01, eta_10 = update_oneind(alphas, betas, self.emissions, p, q)
        p_est = np.exp(eta_01).sum() / np.exp(gammas[0, :-1]).sum()
        q_est = np.exp(eta_10).sum() / np.exp(gammas[1, :-1]).sum()
        return p_est, q_est

    # def update_emi2(self, om1, om2, loglik, max_step_tadmix = 50, max_step_tarchaic = 200, min_interval = 7000):
    #     def update_emi2_oneround():
    #         self.expected_n_coal()
    #         es = self.cache_emissions(version="NCOAL")
    #         es[0] += self.emissions_split[0]
    #         es[1] += self.cache_emissions(z = 1, version="TMRCA")
    #         _, _, _, new_loglik = self.forward_backward_algo(
    #             p=self.p, q=self.q, emissions = es
    #         )
    #         return new_loglik
    #     # update t_admix
    #     old_t_admix = self.t_admix
    #     if np.abs(om1 - self.a1 / self.b1) >= max_step_tadmix:
    #         step_size = np.random.uniform(max_step_tadmix, np.abs(om1 - self.a1 / self.b1))
    #     else:
    #         step_size = np.random.uniform(np.abs(om1 - self.a1 / self.b1), max_step_tadmix)
    #     if om1 > self.a1 / self.b1:
    #         self.t_admix -= step_size
    #     elif om1 < self.a1 / self.b1:
    #         self.t_admix += step_size
    #     new_loglik = update_emi2_oneround()
    #     if new_loglik < loglik:
    #         self.t_admix = old_t_admix
    #     elif self.t_archaic - self.t_admix < min_interval:
    #         self.t_admix = old_t_admix
    # # update t_archaic
    # old_t_archaic = self.t_archaic
    # if np.abs(om2 - self.a2 / self.b2) >= max_step_tarchaic:
    #     step_size = np.random.uniform(max_step_tarchaic, np.abs(om2 - self.a2 / self.b2))
    # else:
    #     step_size = np.random.uniform(np.abs(om2 - self.a2 / self.b2), max_step_tarchaic)
    # if om2 > self.a2 / self.b2:
    #     self.t_archaic -= step_size
    # elif om2 < self.a2 / self.b2:
    #     self.t_archaic += step_size
    # new_loglik = update_emi2_oneround()
    # if new_loglik < loglik:
    #     self.t_archaic = old_t_archaic
    # elif self.t_archaic - self.t_admix < min_interval:
    #     self.t_archaic = old_t_archaic
    # # update emission 2
    # if self.t_admix != old_t_admix or self.t_archaic != old_t_archaic:
    #     self.expected_n_coal()
    #     es = self.cache_emissions(version="NCOAL")
    #     self.emissions_split[2] = es[0]
    #     self.emissions_split[3] = es[1]
    # return
    def baum_welch_ecm(
        self,
        inds,
        niter=10,
        threshold=0.1,
        njobs=4,
        **kwargs,
    ):
        """Implement the Baum-welch algorithm with
        an ECM update step for the mixture of gammas distribution in the non-null case.
        """
        # Setup the accumulators for the parameters ...
        assert niter > 0
        assert len(inds) > 0

        loglik_acc = np.zeros(niter + 1)
        alpha_acc = np.zeros(niter + 1)
        a1_acc = np.zeros(niter + 1)
        a2_acc = np.zeros(niter + 1)
        b1_acc = np.zeros(niter + 1)
        b2_acc = np.zeros(niter + 1)
        p_acc = np.zeros(niter + 1)
        q_acc = np.zeros(niter + 1)
        pi0_acc = np.zeros(niter + 1)
        iter_acc = np.zeros(niter + 1)
        emi_acc = np.zeros(niter + 1)
        tadmix_acc = np.zeros(niter + 1)
        tarchaic_acc = np.zeros(niter + 1)
        alpha_acc[0] = self.alpha
        a1_acc[0] = self.a1
        a2_acc[0] = self.a2
        b1_acc[0] = self.b1
        b2_acc[0] = self.b2
        p_acc[0] = self.p
        q_acc[0] = self.q
        pi0_acc[0] = self.pi0
        tadmix_acc[0] = self.t_admix
        tarchaic_acc[0] = self.t_archaic
        cum_m1 = []
        cum_m2 = []
        om1 = self.a1 / self.b1
        om2 = self.a2 / self.b2

        # Initialize the emissions
        if len(inds) > 1:
            ems = Parallel(n_jobs=njobs)(
                delayed(self.cache_emissions)(ind=ind, z=0) for ind in range(len(inds))
            )
            for j in range(len(inds)):
                self.emissions[inds[j], 0, :] = ems[j]
        else:
            self.emissions_split[0] = self.cache_emissions(z=0, version="TMRCA")
            ems = self.cache_emissions(version="NCOAL")
            self.emissions_split[2] = ems[0]
            self.emissions_split[3] = ems[1]

        for i in tqdm(range(niter)):
            print(self.a1 / self.b1, self.a2 / self.b2)
            print(self.t_admix, self.t_archaic)
            # convergence check
            cum_m1.append(self.a1 / self.b1)
            cum_m2.append(self.a2 / self.b2)
            if i > 3:
                m1 = np.abs(cum_m1[-1] - np.mean(cum_m1[-4:-1]))
                m2 = np.abs(cum_m2[-1] - np.mean(cum_m2[-4:-1]))
                ta = tadmix_acc[i - 1] - tadmix_acc[i - 2]
                tb = tarchaic_acc[i - 1] - tarchaic_acc[i - 2]
                lk = loglik_acc[i - 1] - loglik_acc[i - 2]
                if (
                    lk < threshold
                ):  # or (m1 < threshold and m2 < threshold and ta < threshold and tb < threshold):
                    break
            # update emissions z = 1
            iter_acc[i] = i
            if len(inds) > 1:
                ems = Parallel(n_jobs=njobs)(
                    delayed(self.cache_emissions)(ind=ind, z=1)
                    for ind in range(len(inds))
                )
                for j in range(len(inds)):
                    self.emissions[inds[j], 1, :] = ems[j]
                emi = np.exp(self.emissions[inds, 1, :]) / np.exp(
                    self.emissions[inds, :, :]
                ).sum(axis=1)
                emi_acc[i] = emi[emi > 0.5].size / emi.size
            else:
                self.emissions_split[1] = self.cache_emissions(z=1, version="TMRCA")
                self.combine_emissions()
                emi = np.exp(self.emissions[1]) / np.exp(self.emissions).sum(axis=0)
                emi_acc[i] = emi[emi > 0.5].size / emi.size

            # Get log forward / backward probabilities
            if len(inds) > 1:
                outputs = Parallel(n_jobs=njobs)(
                    delayed(self.forward_backward_algo)(
                        p=self.p, q=self.q, emissions=self.emissions[ind]
                    )
                    for ind in inds
                )
                gammas = np.zeros(shape=(len(inds), 2, self.m))
                alphas = np.zeros(shape=(len(inds), 2, self.m))
                betas = np.zeros(shape=(len(inds), 2, self.m))
                loglik = 0
                for t in range(len(outputs)):
                    gammas[t] = outputs[t][0]
                    alphas[t] = outputs[t][1]
                    betas[t] = outputs[t][2]
                    loglik += outputs[t][3] / len(inds)
                loglik_acc[i] = loglik

                p_gammas = np.exp(gammas[:, 1, :])
                xs_flat = self.xss[p_gammas > 0.5, :].flatten()

                self.alpha, self.a1, self.a2, self.b1, self.b2 = ecm_full_update(
                    xs_flat=xs_flat,
                    alpha=self.alpha,
                    a1=self.a1,
                    a2=self.a2,
                    b1=self.b1,
                    b2=self.b2,
                )
                pi0_hat = np.exp(gammas[:, 0, 0]).mean()
            else:
                gammas, alphas, betas, loglik = self.forward_backward_algo(
                    p=self.p, q=self.q, emissions=self.emissions
                )
                loglik_acc[i] = loglik

                p_gammas = np.exp(gammas[1, :]).reshape(1, gammas.shape[1])
                xs_flat = self.xss[p_gammas[0] > 0.5, :].flatten()

                # EM inference of transitions
                p_est, q_est = self.update_transitions(
                    gammas, alphas, betas, p=self.p, q=self.q, njobs=njobs
                )
                if p_est >= 1e-5 and q_est >= 1e-5:
                    self.p = p_est
                    self.q = q_est

                # update parameters of emission 2
                self.update_emi2(
                    om1=a1_acc[i] / b1_acc[i],
                    om2=a2_acc[i] / b2_acc[i],
                    loglik=loglik_acc[i],
                    max_step_tadmix=100,
                    max_step_tarchaic=500,
                )
                # if self.t_admix != tadmix_acc[i]:
                #     om1 = self.a1 / self.b1
                # if self.t_archaic != tarchaic_acc[i]:
                #     om2 = self.a2 / self.b2

                self.alpha, self.a1, self.a2, self.b1, self.b2 = ecm_full_update(
                    xs_flat=xs_flat,
                    alpha=self.alpha,
                    a1=self.a1,
                    a2=self.a2,
                    b1=self.b1,
                    b2=self.b2,
                )
                pi0_hat = np.exp(gammas[0, 0])

            self.pi0 = pi0_hat

            alpha_acc[i + 1] = self.alpha
            a1_acc[i + 1] = self.a1
            a2_acc[i + 1] = self.a2
            b1_acc[i + 1] = self.b1
            b2_acc[i + 1] = self.b2
            pi0_acc[i + 1] = pi0_hat
            p_acc[i + 1] = self.p
            q_acc[i + 1] = self.q
            tadmix_acc[i + 1] = self.t_admix
            tarchaic_acc[i + 1] = self.t_archaic

        loglik_acc[i] = loglik_acc[-1]
        iter_acc[i] = iter_acc[-1]
        res_dict = {
            "iters": iter_acc,
            "logliks": loglik_acc,
            "a1": a1_acc,
            "a2": a2_acc,
            "b1": b1_acc,
            "b2": b2_acc,
            "alpha": alpha_acc,
            "p": p_acc,
            "q": q_acc,
            "emi": emi_acc,
        }
        return res_dict

    def init_hmm(
        self,
        ts,
        inds,
        njobs,
        recomb_map=False,
        pos_col=1,
        m_col=3,
        alpha=0.3,
        t_admix=None,
        t_archaic=None,
        var_t_admix=None,
        var_t_archaic=None,
        p=0.01,
        q=0.1,
        padmix=0.02,
        ne_archaic=10000,
    ):
        """A wrapper function for HMM initiation."""
        self.add_tree_sequence(ts)
        if recomb_map:
            self.add_recombination_map(recomb_map, pos_col, m_col)
        else:
            self.set_constant_recomb()
        if len(inds) > 1:
            self.xss = np.zeros(shape=(len(inds), self.m, ts.num_samples - 1))
            self.emissions = np.zeros(shape=(ts.num_samples, 2, self.m))
            outputs = Parallel(n_jobs=njobs)(
                delayed(self.extract_tmrca)(i=ind) for ind in inds
            )
            for ind in range(len(inds)):
                self.xss[ind] = outputs[ind]
            self.ncoal = np.zeros(shape=(len(inds), self.m, int(ts.num_samples / 2)))
            outputs = Parallel(n_jobs=njobs)(
                delayed(self.extract_coal_events)(idx=ind) for ind in inds
            )
            for ind in range(len(inds)):
                self.ncoal[ind] = outputs[ind]
        else:
            self.xss = np.zeros(shape=(self.m, ts.num_samples - 1))
            self.emissions = np.zeros(shape=(2, self.m))
            self.emissions_split = np.zeros(shape=(4, self.m))
            self.xss = self.extract_tmrca(i=inds[0])
            self.ncoal = np.zeros(shape=(self.m, int(ts.num_samples / 2)))
            self.ncoal = self.extract_coal_events(idx=inds[0])
        self.est_null_kde()
        self.init_admix_gamma_params(
            alpha=alpha,
            t_admix=t_admix,
            t_archaic=t_archaic,
            var_t_admix=var_t_admix,
            var_t_archaic=var_t_archaic,
            p=p,
            q=q,
        )
        self.padmix = padmix
        self.ne_archaic = ne_archaic
        self.expected_n_coal()

    def train(self, inds, niter=80, njobs=4, seed=1):
        """A wrapper function to train HMM."""
        np.random.seed(seed)
        res_dict = self.baum_welch_ecm(
            inds,
            niter=niter,
            threshold=0.1,
            njobs=njobs,
        )
        return res_dict

    def decode(self, inds, njobs):
        """A wrapper function to decode HMM."""

        def decode_oneind(ind):
            self.xss = self.extract_tmrca(i=ind)
            self.est_null_kde()
            self.emissions = np.zeros(shape=(2, self.m))
            self.emissions_split = np.zeros(shape=(4, self.m))
            # self.set_flattened_emission()
            self.emissions_split[0] = self.cache_emissions(z=0, version="TMRCA")
            self.emissions_split[1] = self.cache_emissions(z=1, version="TMRCA")
            ems = self.cache_emissions(version="NCOAL")
            self.emissions_split[2] = ems[0]
            self.emissions_split[3] = ems[1]
            self.combine_emissions()
            (
                gammas,
                alphas,
                betas,
                _,
            ) = self.forward_backward_algo(p=self.p, q=self.q, emissions=self.emissions)

            return gammas, alphas, betas

        outputs = Parallel(n_jobs=njobs)(
            delayed(decode_oneind)(ind=ind) for ind in inds
        )
        gammas = np.zeros(shape=(len(inds), 2, self.m))
        alphas = np.zeros(shape=(len(inds), 2, self.m))
        betas = np.zeros(shape=(len(inds), 2, self.m))
        for i in range(len(outputs)):
            gammas[i] = outputs[i][0]
            alphas[i] = outputs[i][1]
            betas[i] = outputs[i][2]
        return gammas, alphas, betas


class GhostProductHmm:
    def __init__(self, t_admix=None, t_archaic=None):
        """Initialization of the class."""
        self.ts = None
        self.pos = None
        self.xss = None
        self.f0 = None
        self.emissions = None
        self.emissions_split = None
        self.null_exp_coal = None
        self.intro_exp_coal = None
        self.alpha = 0.5
        self.a1 = None
        self.b1 = None
        self.a2 = None
        self.b2 = None
        self.pi0 = None
        self.p = None
        self.q = None
        self.t_admix = t_admix
        self.t_archaic = t_archaic
        self.padmix = None
        self.ne_archaic = None

    def add_tree_sequence(self, ts):
        """Add in a tree-sequence for analysis.

        The positions extracted are the midpoints of trees.
        """
        self.ts = ts
        assert ts.num_trees > 1
        self.m = ts.num_trees
        self.treespan = np.zeros(shape=(self.m, 2))
        self.pos = np.zeros(self.m)
        for i, t in enumerate(ts.trees()):
            self.pos[i] = (t.interval.right + t.interval.left) / 2.0
            self.treespan[i] = np.array([t.interval.left, t.interval.right])
        self.pi0 = 0.5

    def extract_tmrca(self, i, js=None, seed=1):
        """Extract TMRCA as the primary feature for downstream modeling.

        The (M x (n-1))
        """
        assert self.ts is not None
        assert i < self.ts.num_samples
        if js is None:
            js = np.array([x for x in self.ts.samples()])
        assert js.size > 0
        out_other = np.array(
            [[tree.tmrca(i, j) for j in js if i != j] for tree in self.ts.trees()]
        )
        np.random.seed(seed)
        out_self = np.random.uniform(0, 1, size=(out_other.shape[0], 1))
        output = np.concatenate((out_self, out_other), axis=1)
        assert output.shape[1] == len(js) or output.shape[1] == len(js) + 1
        assert output.shape[0] == self.ts.num_trees
        return output

    def est_ne(self, t0=1000, t1=2000, ne_max=1e5, step=100):
        """Estimate the effective population size within the interval."""
        assert self.ts is not None
        assert self.ts.num_samples > 0
        assert t0 < t1
        ave_nt0 = 0
        ave_nt1 = 0
        tt0 = t0
        tt1 = t1
        while ave_nt0 == ave_nt1:
            ts0 = self.ts.decapitate(tt0)
            ts1 = self.ts.decapitate(tt1)
            nt0 = np.array([tx.num_roots for (_, _, tx) in self.ts.coiterate(ts0)])
            nt1 = np.array([tx.num_roots for (_, _, tx) in self.ts.coiterate(ts1)])
            ave_nt0 = int(np.mean(nt0))
            ave_nt1 = int(np.mean(nt1))
            tt0 -= 100
            tt1 += 100
        nes = []
        for ne in np.arange(100, ne_max, step):
            Q = DistinguishedLineages().create_Q(ave_nt0)
            P = expm(Q * (t1 - t0) / (2 * ne))
            nes.append(P[0, ave_nt0 - ave_nt1])
        return np.arange(100, ne_max, step)[np.argmax(nes)]

    def expected_n_coal(self):
        """Expected number of coalescent events in this interval.

        NOTE: t2 > t1, so we expect more lineages to be in t1's range ...
        """
        assert self.ncoal is not None
        assert self.padmix is not None
        assert self.ne_archaic is not None
        ts1 = self.ts.decapitate(self.t_admix)
        ts2 = self.ts.decapitate(self.t_archaic)
        nt1 = np.array([tx.num_roots for (_, _, tx) in self.ts.coiterate(ts1)])
        nt2 = np.array([tx.num_roots for (_, _, tx) in self.ts.coiterate(ts2)])
        ne_tadmix = self.est_ne(t0=self.t_admix - 100, t1=self.t_admix + 100)
        self.null_exp_coal = exp_ncoal_null(nt1=nt1, nt2=nt2)
        self.intro_exp_coal = exp_ncoal_intro(
            nt1=np.mean(nt1),
            ne=ne_tadmix,
            t_interval=self.t_archaic - self.t_admix,
            padmix=self.padmix,
            ne_archaic=self.ne_archaic,
        )

    def extract_coal_events(self, idx=0, t1=100, t2=1000):
        """Method to extract the tmrcas of coalescent events features for the HMM.
        Arguments:

        - ts: tskit TreeSequence containing inferred ARG
        - idx: haplotype index of test haplotypes
        - t1: time in generations for lower bound (tadmix)
        - t2: time in generations for upper bound (tarchaic)

        """
        assert t2 >= t1
        assert self.ts is not None
        assert self.ts.num_samples > 0
        n = self.ts.num_trees
        ncoal = np.zeros(shape=(n, int(self.ts.num_samples - 1)))
        for i, tree in enumerate(self.ts.trees()):
            tx = 0
            k = idx
            while k != tskit.NULL:
                ncoal[i, tx] = tree.time(k)
                k = tree.parent(k)
                tx += 1
        return ncoal

    def extract_n_coal(self):
        """Extract the number of coalescent events in the interval."""
        assert self.ts is not None
        assert self.t_admix is not None
        assert self.t_archaic is not None
        assert self.ncoal is not None
        assert self.ncoal.shape[0] == self.m
        assert self.ncoal.shape[1] == int(self.ts.num_samples - 1)
        temp = np.copy(self.ncoal, order="C")
        ncoal = extract_n_coal_cython(
            ncoal=temp, tadmix=self.t_admix, tarchaic=self.t_archaic
        )
        return ncoal

    def add_recombination_map(self, recmap, pos_col=1, m_col=3):
        """Add and interpolate a recombination rate between every location.

        Recmap: a recombination map file.
        pos_col: column number for physical positions (0-index).
        m_col: column number for genetic distances (in Morgan, 0-index).
        """
        assert self.ts is not None
        df = pd.read_csv(recmap, sep="\s+")
        recmap = df.iloc[:, [pos_col, m_col]].to_numpy().astype("float")
        assert np.all(np.diff(recmap[:, 0]) > 0)
        assert np.all(np.diff(recmap[:, 1]) > 0)
        interp_recmap = interp1d(recmap[:, 0], recmap[:, 1])
        self.recmap = interp_recmap(self.pos)

    def set_constant_recomb(self, rate=1e-8):
        """Set a constant recombination rate per basepair per generation."""
        assert self.ts is not None
        assert self.pos.size > 1
        assert (rate > 0.0) and (rate < 1.0)
        self.recmap = rate * self.pos
        self.treespan = self.treespan / 1e6

    def est_null_kde(self, i=False, npts=10000):
        """Estimate null KDEs via interpolation."""
        assert npts > 100
        # assert np.all(self.xss > 0.0)
        assert self.xss is not None
        if i:
            x_prime = self.xss[i].flatten()
        else:
            x_prime = self.xss.flatten()
        fft_kde = FFTKDE(bw="silverman").fit(data=x_prime)
        grid_eval = np.linspace(np.min(x_prime) - 1, np.max(x_prime) + 1, npts)
        y_fft_kde = fft_kde.evaluate(grid_eval)
        f_kde = interp1d(grid_eval, y_fft_kde, kind="quadratic", bounds_error=False)
        self.f0 = f_kde

    def cache_emissions(self, version="NCOAL", z=1):
        """Create a vector-based cache of emissions for speed."""
        assert self.f0 is not None
        assert self.xss is not None
        # assert self.ncoal is not None
        # assert self.t_admix is not None
        # assert self.t_archaic is not None
        # assert self.intro_exp_coal is not None
        # assert self.null_exp_coal is not None

        if version == "NCOAL":
            emission = np.zeros(shape=(2, self.m))
            tncoal = self.extract_n_coal()
            emission[0] = emission_coal_event(
                x=tncoal, z=0, lamb_n=self.null_exp_coal, lamb0=self.intro_exp_coal
            )
            emission[1] = emission_coal_event(
                x=tncoal, z=1, lamb_n=self.null_exp_coal, lamb0=self.intro_exp_coal
            )
        if version == "TMRCA":
            emission = np.zeros(self.m)
            if z == 0:
                for i in range(self.m):
                    emission[i] = np.log(self.f0(self.xss[i, :])).mean()
            else:
                emission = emission_product_z1(
                    self.xss, self.alpha, self.a1, self.b1, self.a2, self.b2
                )
        return emission

    def combine_emissions(self):
        assert self.emissions_split is not None
        self.emissions[0] = self.emissions_split[0] + self.emissions_split[2]
        self.emissions[1] = self.emissions_split[1] + self.emissions_split[3]

    def find_crossing_point(self, f=None, g=None, mu1=50e3):
        """Find the crossing point that is above the initial distribution."""
        assert f is not None
        assert g is not None
        xs = np.linspace(0.0, np.max(self.xss), 1000)
        idx = np.argwhere(np.diff(np.sign(f(xs) - g(xs)))).flatten()
        cp = xs[idx]
        if len(cp[cp > mu1]) > 0:
            cp_prime = np.min(cp[cp > mu1])
        else:
            cp_prime = np.max(self.xss) - 1
        return idx, xs, cp, cp_prime

    def double_flatten(self, f=None, g=None, x_prime=54e3, alpha=1e-2):
        """Apply a double flattening of the"""
        assert x_prime > np.min(self.xss)
        assert x_prime <= np.max(self.xss)
        assert (alpha < 1.0) & (alpha > 0)
        interval = np.max(self.xss) - x_prime
        af, _ = quad(func=f, a=np.min(self.xss), b=x_prime)
        ag, _ = quad(func=g, a=np.min(self.xss), b=x_prime)
        f_prime = (
            lambda x: ((1 - alpha / 2) / af) * f(x)
            if x <= x_prime
            else (alpha / 2) / interval
        )
        g_prime = (
            lambda x: ((1 - alpha) / ag) * g(x) if x <= x_prime else (alpha) / interval
        )
        return f_prime, g_prime, interval, alpha, af, ag

    def cache_emissions_flat(
        self, f=None, x_prime=None, a=1e-2, interval=None, integrand=None
    ):
        """Cache the flattened emission model prior to full inference using the forward-backward algorithm."""
        assert self.xss is not None
        assert f is not None
        assert x_prime is not None
        assert self.ncoal is not None

        emission0 = np.zeros(self.m)
        input = self.xss
        for i in range(self.m):
            emission0[i] = np.log(f(input[i, :])).mean()
        emission1 = emission_product_z1_flat(
            input,
            self.alpha,
            self.a1,
            self.b1,
            self.a2,
            self.b2,
            x_prime=x_prime,
            a=a,
            interval=interval,
            integrand=integrand,
        )
        return emission0, emission1

    def set_flattened_emission(self, alpha=1e-2):
        """Run the full update for the emission?"""
        assert self.xss is not None
        g1 = lambda x: self.alpha * gamma_sp.pdf(x, a=self.a1, scale=1 / self.b1) + (
            1 - self.alpha
        ) * gamma_sp.pdf(x, a=self.a2, scale=1 / self.b2)
        _, _, _, cp = self.find_crossing_point(f=self.f0, g=g1, mu1=self.a2 / self.b2)
        print(cp)
        f_prime, g_prime, interval, alpha, af, ag = self.double_flatten(
            f=self.f0, g=g1, x_prime=cp, alpha=alpha
        )
        f_prime = np.vectorize(f_prime)
        g_prime = np.vectorize(g_prime)
        e0, e1 = self.cache_emissions_flat(
            f=f_prime, x_prime=cp, a=alpha / 2, interval=interval, integrand=ag
        )
        self.emissions[0, :] = e0
        self.emissions[1, :] = e1
        self.combine_emissions()
        return f_prime, g_prime

    def init_admix_gamma_params(
        self,
        alpha=0.03,
        t_admix=None,
        t_archaic=None,
        var_t_admix=None,
        var_t_archaic=None,
        p=0.01,
        q=0.1,
        null_data=None,
    ):
        """Initialize the parameters for the non-null mixture distribution."""
        if null_data is None:
            null_data = self.xss
        if t_admix is None:
            t_admix = 2000
        if t_archaic is None:
            t_archaic = np.percentile(null_data.flatten(), 80)
        self.t_admix = t_admix
        self.t_archaic = t_archaic
        t2 = np.mean(null_data[null_data > t_archaic])
        t1 = np.mean(null_data[null_data < t_admix])
        if var_t_admix is None:
            var_t_admix = np.var(null_data[null_data < t_admix].flatten())
        if var_t_archaic is None:
            var_t_archaic = np.var(null_data[null_data > t_archaic].flatten())
        self.alpha = alpha
        self.b1 = t1 / var_t_admix
        self.a1 = (t1**2) / var_t_admix
        self.b2 = t2 / var_t_archaic
        self.a2 = (t2**2) / var_t_archaic
        self.p = p
        self.q = q

    def forward_algo(self, p=1e-2, q=1e-2, emissions=None):
        """Implement the forward algorithm for the binary hmm."""
        assert (p > 0) and (q > 0)
        assert (self.a1 > 0) and (self.b1 > 0)
        assert (self.a2 > 0) and (self.b2 > 0)
        assert self.pos.size == self.m
        assert self.emissions is not None
        es = np.copy(emissions, order="C")
        alphas, scaler, loglik = forward_algo_product(
            p=p,
            q=q,
            es=es,
            pi0=self.pi0,
        )
        return alphas, scaler, loglik

    def backward_algo(self, p=1e-2, q=1e-2, emissions=None):
        assert (p > 0) and (q > 0)
        assert (self.a1 > 0) and (self.b1 > 0)
        assert (self.a2 > 0) and (self.b2 > 0)
        assert self.pos.size == self.m
        assert self.emissions is not None
        es = np.copy(emissions, order="C")
        betas, scaler, loglik = backward_algo_product(
            p=p,
            q=q,
            es=es,
        )
        return betas, scaler, loglik

    def forward_backward_algo(self, **kwargs):
        """Forward-backward algorithm implementation.

        Returns only the gamma values for the marginal posteriors
        """
        alphas, _, loglik_fwd = self.forward_algo(**kwargs)
        betas, _, loglik_bwd = self.backward_algo(**kwargs)
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, alphas, betas, loglik_fwd + loglik_bwd

    def update_transitions(self, gammas, alphas, betas, p, q):
        """Updates for the transitions."""
        assert alphas.size == betas.size
        assert (p > 0) and (q > 0)
        assert (p < 1) and (q < 1)
        m = self.m
        eta_01 = np.zeros(m - 1)
        eta_10 = np.zeros(m - 1)

        def update_oneind(alphas, betas, emissions, p, q):  # , ind = None):
            eta01, eta10 = update_oneind_cython(
                m=m,
                alphas=alphas,
                betas=betas,
                emissions=emissions,
                p=p,
                q=q,
            )
            return eta01, eta10

        eta_01, eta_10 = update_oneind(alphas, betas, self.emissions, p, q)
        p_est = np.exp(eta_01).sum() / np.exp(gammas[0, :-1]).sum()
        q_est = np.exp(eta_10).sum() / np.exp(gammas[1, :-1]).sum()
        return p_est, q_est

    def baum_welch_ecm(
        self,
        niter=10,
        threshold=0.1,
        **kwargs,
    ):
        """Implement the Baum-welch algorithm with
        an ECM update step for the mixture of gammas distribution in the non-null case.
        """
        # Setup the accumulators for the parameters ...
        assert niter > 0

        loglik_acc = np.zeros(niter + 1)
        alpha_acc = np.zeros(niter + 1)
        a1_acc = np.zeros(niter + 1)
        a2_acc = np.zeros(niter + 1)
        b1_acc = np.zeros(niter + 1)
        b2_acc = np.zeros(niter + 1)
        p_acc = np.zeros(niter + 1)
        q_acc = np.zeros(niter + 1)
        pi0_acc = np.zeros(niter + 1)
        iter_acc = np.zeros(niter + 1)
        alpha_acc[0] = self.alpha
        a1_acc[0] = self.a1
        a2_acc[0] = self.a2
        b1_acc[0] = self.b1
        b2_acc[0] = self.b2
        p_acc[0] = self.p
        q_acc[0] = self.q
        pi0_acc[0] = self.pi0
        cum_m1 = []
        cum_m2 = []

        # Initialize the emissions
        self.emissions_split[0] = self.cache_emissions(z=0, version="TMRCA")
        # ems = self.cache_emissions(version="NCOAL")
        # self.emissions_split[2] = ems[0]
        # self.emissions_split[3] = ems[1]

        for i in tqdm(range(niter)):
            if i > 3:
                lk = loglik_acc[i - 1] - loglik_acc[i - 2]
                if (
                    lk < threshold
                ):  # or (m1 < threshold and m2 < threshold and ta < threshold and tb < threshold):
                    break
            # update emissions z = 1
            iter_acc[i] = i
            self.emissions_split[1] = self.cache_emissions(z=1, version="TMRCA")
            self.combine_emissions()
            gammas, alphas, betas, loglik = self.forward_backward_algo(
                p=self.p, q=self.q, emissions=self.emissions
            )
            loglik_acc[i] = loglik

            p_gammas = np.exp(gammas[1, :]).reshape(1, gammas.shape[1])
            xs_flat = self.xss[p_gammas[0] > 0.5, :].flatten()

            self.alpha, self.a1, self.a2, self.b1, self.b2 = ecm_full_update(
                xs_flat=xs_flat,
                alpha=self.alpha,
                a1=self.a1,
                a2=self.a2,
                b1=self.b1,
                b2=self.b2,
            )
            pi0_hat = np.exp(gammas[0, 0])
            self.pi0 = pi0_hat

            # EM inference of transitions
            p_est, q_est = self.update_transitions(
                gammas, alphas, betas, p=self.p, q=self.q
            )
            if p_est >= 1e-5 and q_est >= 1e-5:
                self.p = p_est
                self.q = q_est

            alpha_acc[i + 1] = self.alpha
            a1_acc[i + 1] = self.a1
            a2_acc[i + 1] = self.a2
            b1_acc[i + 1] = self.b1
            b2_acc[i + 1] = self.b2
            pi0_acc[i + 1] = pi0_hat
            p_acc[i + 1] = self.p
            q_acc[i + 1] = self.q

        res_dict = {
            "iters": iter_acc,
            "logliks": loglik_acc,
            "a1": a1_acc,
            "a2": a2_acc,
            "b1": b1_acc,
            "b2": b2_acc,
            "alpha": alpha_acc,
            "p": p_acc,
            "q": q_acc,
        }
        return res_dict

    def prepare_data_tmrca(self, ts, ind, js=None):
        """A wrapper function to prepare data for HMM."""
        self.ts = ts
        self.xss = self.extract_tmrca(i=ind, js=js)
        self.est_null_kde()
        return self.xss, self.f0

    def init_hmm(
        self,
        ts,
        ind,
        xss,
        f0,
        recomb_map=False,
        pos_col=1,
        m_col=3,
        alpha=0.3,
        t_admix=None,
        t_archaic=None,
        var_t_admix=None,
        var_t_archaic=None,
        p=0.01,
        q=0.1,
        padmix=0.02,
        ne_archaic=10000,
    ):
        """A wrapper function for HMM initiation."""
        self.add_tree_sequence(ts)
        if recomb_map:
            self.add_recombination_map(recomb_map, pos_col, m_col)
        else:
            self.set_constant_recomb()
        self.xss = xss
        self.ncoal = self.extract_coal_events(idx=ind)
        self.f0 = f0
        self.emissions = np.zeros(shape=(2, self.m))
        self.emissions_split = np.zeros(shape=(4, self.m))
        self.init_admix_gamma_params(
            alpha=alpha,
            t_admix=t_admix,
            t_archaic=t_archaic,
            var_t_admix=var_t_admix,
            var_t_archaic=var_t_archaic,
            p=p,
            q=q,
        )
        self.padmix = padmix
        self.ne_archaic = ne_archaic
        # self.expected_n_coal()

    def train(self, niter=80, seed=1, threshold=0.1):
        """A wrapper function to train HMM."""
        np.random.seed(seed)
        res_dict = self.baum_welch_ecm(
            niter=niter,
            threshold=threshold,
        )
        return res_dict

    def decode(self):
        """A wrapper function to decode HMM."""
        # self.set_flattened_emission()
        # self.xss = self.extract_tmrca(i=ind)
        # self.est_null_kde()
        # self.emissions = np.zeros(shape=(2, self.m))
        # self.emissions_split = np.zeros(shape=(4, self.m))
        # self.emissions_split[0] = self.cache_emissions(z = 0, version="TMRCA")
        self.emissions_split[1] = self.cache_emissions(z=1, version="TMRCA")
        # ems = self.cache_emissions(version="NCOAL")
        # self.emissions_split[2] = ems[0]
        # self.emissions_split[3] = ems[1]
        self.combine_emissions()
        (
            gammas,
            alphas,
            betas,
            _,
        ) = self.forward_backward_algo(p=self.p, q=self.q, emissions=self.emissions)
        return gammas, alphas, betas

    def params_reestimate(self, p_gammas, xss_selfpop, t_admix):
        """Re-estimate the parameters.
        p_gammas: exponentiated gamma values / states for alternative state.
        """
        xs_flat = xss_selfpop[p_gammas > 0.5, :].flatten()
        # init the parameters
        t2 = np.mean(xs_flat[xs_flat > t_admix])
        t1 = np.mean(xs_flat[xs_flat < t_admix])
        var_t_admix = np.var(xs_flat[xs_flat < t_admix].flatten())
        var_t_archaic = np.var(xs_flat[xs_flat > t_admix].flatten())
        b1 = t1 / var_t_admix
        a1 = (t1**2) / var_t_admix
        b2 = t2 / var_t_archaic
        a2 = (t2**2) / var_t_archaic
        d1 = Gamma([a1], [b1])
        d2 = Gamma([a2], [b2])
        model = GeneralMixtureModel([d1, d2]).fit(
            xs_flat.reshape([xs_flat.shape[0], 1])
        )
        return d1.shapes[0], d1.rates[0], d2.shapes[0], d2.rates[0], model.priors[0]
