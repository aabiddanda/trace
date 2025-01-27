"""Implementation of an HMM to detect ghost admixture."""
import numpy as np
import pandas as pd
import tskit
from joblib import Parallel, delayed
from KDEpy import FFTKDE
from pomegranate.distributions import *
from pomegranate.gmm import GeneralMixtureModel
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.linalg import expm
from scipy.special import logsumexp as logsumexp_sp
from scipy.stats import binom
from scipy.stats import gamma as gamma_sp
from tqdm import tqdm

from ghosthmm_utils import (
    backward_algo_product,
    ecm_full_update,
    emission_product_z1,
    emission_product_z1_flat,
    forward_algo_product,
    update_oneind_cython,
)


class GhostProductHmm:
    def __init__(self, t_admix=None, t_archaic=None):
        """Initialization of the class."""
        self.ts = None
        self.pos = None
        self.xss = None
        self.f0 = None
        self.emissions = None
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

    def add_tree_sequence(self, ts, subrange=None):
        """Add in a tree-sequence for analysis.

        The positions extracted are the midpoints of trees.
        """
        self.ts = ts
        assert ts.num_trees > 1
        if subrange is None:
            left_edge = 0
            right_edge = self.ts.sequence_length
        else:
            left_edge = subrange[0]
            right_edge = subrange[1]
        self.m = ts.num_trees
        self.treespan = np.zeros(shape=(self.m, 2))
        self.pos = np.zeros(self.m)
        for i, t in enumerate(ts.trees()):
            if t.interval.left >= right_edge:
                break
            elif t.interval.left < left_edge:
                continue
            else:
                self.pos[i] = (t.interval.right + t.interval.left) / 2.0
                self.treespan[i] = np.array([t.interval.left, t.interval.right])
        self.treespan = self.treespan[self.treespan[:, 1] > 0]
        self.treespan_phy = self.treespan.copy()
        self.pos = self.pos[self.pos > 0]
        self.m = self.pos.size
        self.pi0 = 0.5

    def extract_tmrca(self, i, js=None, subrange=None, seed=1):
        """Extract TMRCA as the primary feature for downstream modeling.

        The (M x (n-1))
        """
        assert self.ts is not None
        assert i < self.ts.num_samples
        if js is None:
            js = np.array([x for x in self.ts.samples()])
        assert js.size > 0
        if subrange is None:
            left_edge = 0
            right_edge = self.ts.sequence_length
        else:
            left_edge = subrange[0]
            right_edge = subrange[1]
        out_other = []
        for tree in self.ts.trees():
            if tree.interval.left >= right_edge:
                break
            elif tree.interval.left < left_edge:
                continue
            else:
                out_other.append([tree.tmrca(i, j) for j in js if i != j])
        out_other = np.array(out_other)
        np.random.seed(seed)
        out_self = np.random.uniform(0, 1, size=(out_other.shape[0], 1))
        output = np.concatenate((out_self, out_other), axis=1)
        assert output.shape[1] == len(js) or output.shape[1] == len(js) + 1
        return output

    def add_recombination_map(self, recmap, pos_col=1, m_col=3):
        """Add and interpolate a recombination rate between every location.

        Recmap: a recombination map file.
        pos_col: column number for physical positions (0-index).
        m_col: column number for genetic distances (in Centi-Morgan (cM), 0-index).
        """
        assert self.pos is not None
        assert self.treespan is not None
        df = pd.read_csv(recmap, sep="\s+")
        recmap = df.iloc[:, [pos_col, m_col]].to_numpy().astype("float")
        if not recmap[0, 0] == 0.0:
            recmap = np.insert(recmap, 0, [0.0, 0.0], axis=0)
        interp_recmap = interp1d(recmap[:, 0], recmap[:, 1])
        self.pos = interp_recmap(self.pos)
        self.treespan = interp_recmap(self.treespan)

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

    def cache_emissions(self, z=1):
        """Create a vector-based cache of emissions for speed."""
        assert self.f0 is not None
        assert self.xss is not None

        emission = np.zeros(self.m)
        if z == 0:
            for i in range(self.m):
                emission[i] = np.log(self.f0(self.xss[i, :])).mean()
        else:
            emission = emission_product_z1(
                self.xss, self.alpha, self.a1, self.b1, self.a2, self.b2
            )
        return emission

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
        if len(null_data[null_data > t_archaic]) == 0:
            t2 = t_archaic
        t1 = np.mean(null_data[null_data < t_admix])
        if len(null_data[null_data < t_admix]) == 0:
            t1 = t_admix
        if var_t_admix is None:
            var_t_admix = np.var(null_data[null_data < t_admix].flatten())
            i = 1
            temp = t_admix
            while len(null_data[null_data < temp]) == 0 or var_t_admix == 0:
                temp = np.percentile(null_data.flatten(), 100 - 10 * i)
                var_t_admix = np.var(null_data[null_data < temp].flatten())
                i += 1
        if var_t_archaic is None:
            var_t_archaic = np.var(null_data[null_data >= t_archaic].flatten())
            i = 1
            temp = t_archaic
            while len(null_data[null_data >= temp]) == 0 or var_t_archaic == 0:
                temp = np.percentile(null_data.flatten(), 100 - 10 * i)
                var_t_archaic = np.var(null_data[null_data >= temp].flatten())
                i += 1
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

        # Initialize the emissions
        self.emissions[0] = self.cache_emissions(z=0)

        for i in tqdm(range(niter)):
            if i > 3:
                lk = loglik_acc[i - 1] - loglik_acc[i - 2]
                if lk < threshold:
                    break
            # update emissions z = 1
            iter_acc[i] = i
            self.emissions[1] = self.cache_emissions(z=1)
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

    def prepare_data_tmrca(self, ts, ind, subrange=None, js=None):
        """A wrapper function to prepare data for HMM."""
        self.add_tree_sequence(ts, subrange=subrange)
        self.xss = self.extract_tmrca(i=ind, js=js, subrange=subrange)
        self.est_null_kde()
        return self.xss, self.f0

    def init_hmm(
        self,
        ts,
        xss,
        f0,
        subrange=None,
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
    ):
        """A wrapper function for HMM initiation."""
        self.add_tree_sequence(ts, subrange=subrange)
        self.xss = xss
        self.f0 = f0
        if recomb_map:
            self.add_recombination_map(recomb_map, pos_col, m_col)
        else:
            self.set_constant_recomb()
        self.emissions = np.zeros(shape=(2, self.m))
        self.init_admix_gamma_params(
            alpha=alpha,
            t_admix=t_admix,
            t_archaic=t_archaic,
            var_t_admix=var_t_admix,
            var_t_archaic=var_t_archaic,
            p=p,
            q=q,
        )

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
        self.emissions[1] = self.cache_emissions(z=1)
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
