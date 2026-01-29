"""Implementation of an HMM to detect ghost admixture."""
import numpy as np
import pandas as pd
import pybedtools
import tskit
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.special import digamma
from scipy.special import logsumexp as logsumexp_sp
from scipy.stats import gamma as gamma_sp
from scipy.stats import t as t_sp
from tqdm import tqdm
from trace_utils import (
    backward_algo_product,
    extract_n_coal_cython,
    forward_algo_product,
    gamma_logpdf,
    logsumexp,
    update_oneind_cython,
)


class TRACE:
    """Class defining the TRACE HMM Model."""

    def __init__(self, t_admix=None, t_archaic=None):
        """Initialize the class."""
        self.ts = None
        self.emissions = None
        self.a1 = None
        self.b1 = None
        self.a2 = None
        self.b2 = None
        self.pi0 = 0.5
        self.p = None
        self.q = None

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
        for i, t in enumerate(ts.trees()):
            if t.interval.left >= right_edge:
                break
            elif t.interval.left < left_edge:
                continue
            else:
                self.treespan[i] = np.array([t.interval.left, t.interval.right])
        self.treespan = self.treespan[self.treespan[:, 1] > 0]

    def mask_regions(self, treespan, chrom, maskfile, f=None):
        """Mask regions of the treespan."""
        assert self.treespan is not None
        assert maskfile is not None
        self.mask = np.ones(treespan.shape[0])
        mask = pybedtools.BedTool(maskfile)
        treespan = treespan.astype(int)
        rows = [f"{chrom}\t{start}\t{end}" for start, end in treespan]
        tspan = "\n".join(rows) + "\n"
        tspan = pybedtools.BedTool(tspan, from_string=True)
        if f is not None:
            intersects = tspan.intersect(mask, u=True, f=f)
        else:
            intersects = tspan.intersect(mask, u=True)
        overlapping = set((i.chrom, i.start, i.end) for i in intersects)
        for i in range(self.treespan.shape[0]):
            if (
                not (chrom, int(self.treespan[i][0]), int(self.treespan[i][1]))
                in overlapping
            ):
                self.mask[i] = 0
        return self.mask

    def extract_ncoal(self, idx, t_archaic, subrange=None):
        """Extract the number of coalescent events outside the focal branch."""
        if subrange is None:
            left_edge = 0
            right_edge = self.ts.sequence_length
        else:
            left_edge = subrange[0]
            right_edge = subrange[1]
        # check t_archaic, break the code into multiple rows for better readability
        assert 0 < t_archaic < self.ts.max_root_time, (
            f"t_archaic must be between 0 and {self.ts.max_root_time} "
            f"(max tree height in the tree sequence)."
        )
        self.ncoal = []
        self.t1s = []
        self.t2s = []
        self.n_leaves = []
        if isinstance(idx, int):
            idx = [idx]
        for i in range(len(idx)):
            self.ncoal.append([])
            self.t1s.append([])
            self.t2s.append([])
            self.n_leaves.append([])
        for tree in self.ts.trees():
            if tree.interval.left >= right_edge:
                break
            elif tree.interval.left < left_edge:
                continue
            else:
                ns = []
                for node in tree.nodes():
                    ns.append(tree.time(node))
                ns = np.array(ns)
                for id_index, id in enumerate(idx):
                    if id not in tree.samples():
                        raise ValueError(f"Error: sample {id} is not a leaf node.")
                    xsfs = 0
                    nls = 0
                    t1 = 0
                    t2 = 0
                    ncoal = []
                    nos = []
                    k = id
                    while k != tskit.NULL:
                        ncoal.append(tree.time(k))
                        nos.append(k)
                        k = tree.parent(k)
                    for i in range(len(ncoal) - 1):
                        t1 = ncoal[i]
                        t2 = ncoal[i + 1]
                        if t1 <= t_archaic and t2 > t_archaic:
                            xsfs = extract_n_coal_cython(
                                ncoal=ns, tadmix=t1, tarchaic=t2
                            )
                            nls = tree.num_samples(nos[i])
                            break
                    if t2 <= t_archaic:
                        self.t1s[id_index].append(t2)
                        if t2 == 0:
                            self.n_leaves[id_index].append(1)
                        else:
                            self.n_leaves[id_index].append(tree.num_samples())
                    else:
                        self.t1s[id_index].append(t1)
                        self.n_leaves[id_index].append(nls)
                    self.t2s[id_index].append(t2)
                    self.ncoal[id_index].append(xsfs)
        self.ncoal = np.array(self.ncoal)
        self.t1s = np.array(self.t1s)
        self.t2s = np.array(self.t2s)
        self.ncoal = self.ncoal * (self.t2s - self.t1s) + 1e-10
        self.n_leaves = np.array(self.n_leaves)
        if len(idx) == 1:
            self.ncoal = self.ncoal[0]
            self.t1s = self.t1s[0]
            self.t2s = self.t2s[0]
            self.n_leaves = self.n_leaves[0]
        return self.ncoal, self.t1s, self.t2s, self.n_leaves

    def add_recombination_map(self, treespan, recmap, pos_col=1, m_col=3, skiprow=True):
        """Add and interpolate a recombination rate between every location.

        Recmap: a recombination map file.
        pos_col: column number for physical positions (0-index).
        m_col: column number for genetic distances (in Centi-Morgan (cM), 0-index).
        """
        df = pd.read_csv(recmap, sep="\s+", header=None, skiprows=int(skiprow))
        recmap = df.iloc[:, [pos_col, m_col]].to_numpy().astype("float")
        if not recmap[0, 0] == 0.0:
            recmap = np.insert(recmap, 0, [0.0, 0.0], axis=0)
        if not recmap[-1, 0] >= np.max(treespan[:, 1]):
            recmap = np.insert(
                recmap, recmap.shape[0], [np.max(treespan[:, 1]), recmap[-1, 1]], axis=0
            )
        interp_recmap = interp1d(recmap[:, 0], recmap[:, 1])
        treespan = interp_recmap(treespan)
        return treespan

    def set_constant_recomb(self, rate=1e-8):
        """Set a constant recombination rate per basepair per generation."""
        assert (rate > 0.0) and (rate < 1.0)
        self.treespan = self.treespan / 1e6

    def cache_emissions(self, z=1, alpha=1e-2):
        """Create a vector-based cache of emissions for speed."""
        assert self.ncoal is not None
        assert self.emi2_a2 is not None
        assert self.emi2_b2 is not None
        assert self.mask is not None
        emission = np.zeros(self.ncoal.shape)
        if z == 1:
            # check if tail is flipped
            f = lambda x: gamma_sp.pdf(x, a=self.emi2_a1, scale=1 / self.emi2_b1)
            g = lambda x: gamma_sp.pdf(x, a=self.emi2_a2, scale=1 / self.emi2_b2)
            _, _, _, cp = self.find_crossing_point(
                f=f, g=g, mu1=self.emi2_a2 / self.emi2_b2
            )
            if cp is not None:
                interval = np.max(self.ncoal) - cp
                gl = gamma_sp.cdf(cp, a=self.emi2_a2, scale=1 / self.emi2_b2)
                fl = gamma_sp.cdf(cp, a=self.emi2_a1, scale=1 / self.emi2_b1)
                sl = (fl - alpha) / gl
                for i in range(self.m):
                    if self.mask[i] == 0:
                        emission[i] = np.log(1)
                    else:
                        if self.ncoal[i] >= cp:
                            emission[i] = gamma_logpdf(
                                self.ncoal[i], self.emi2_a1, self.emi2_b1
                            ) + np.log(alpha / interval)
                        else:
                            emission[i] = gamma_logpdf(
                                self.ncoal[i], self.emi2_a2, self.emi2_b2
                            ) + np.log(sl)
            else:
                for i in range(self.m):
                    if self.mask[i] == 0:
                        emission[i] = np.log(1)
                    else:
                        emission[i] = gamma_logpdf(
                            self.ncoal[i], self.emi2_a2, self.emi2_b2
                        )
        else:
            for i in range(self.m):
                if self.mask[i] == 0:
                    emission[i] = np.log(1)
                else:
                    emission[i] = gamma_logpdf(
                        self.ncoal[i], self.emi2_a1, self.emi2_b1
                    )
        emission = emission + np.log(self.scale_factor)
        return emission

    def find_crossing_point(self, f=None, g=None, mu1=50e3):
        """Find the crossing point that is above the initial distribution."""
        assert f is not None
        assert g is not None
        xs = np.linspace(0.0, np.max(self.ncoal), 1000)
        idx = np.argwhere(np.diff(np.sign(f(xs) - g(xs)))).flatten()
        cp = xs[idx]
        if len(cp[cp > mu1]) > 0:
            cp_prime = np.min(cp[cp > mu1])
        else:
            cp_prime = None
        return idx, xs, cp, cp_prime

    def generalized_esd_test(self, data=None, alpha=0.05, max_outlier_prop=0.1):
        """Generalized ESD test for identifying residual outliers."""
        if data is None:
            data = self.ncoal
        else:
            data = np.array(data)
        n = data.size
        max_outliers = max(1, int(n * max_outlier_prop))
        outliers = []
        for i in range(1, max_outliers + 1):
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            residuals = data - mean  # modified to be one-sided, so not taking abs
            max_residual_idx = np.argmax(residuals)  # would return the first index
            max_residual = residuals[max_residual_idx]
            # Test statistic max(|X_i - X_bar|) / s
            R = max_residual / std
            # Compute critical value
            p = 1 - alpha / (n - i + 1)  # Bonferroni correction, one-sided
            t_critical = t_sp.ppf(p, df=n - i - 1)
            lambda_critical = (
                (n - i)
                * t_critical
                / np.sqrt((n - i - 1 + t_critical**2) * (n - i + 1))
            )
            if R > lambda_critical:
                outliers.append(data[max_residual_idx])
                data = np.delete(data, max_residual_idx)
            else:
                break
        return outliers, data

    def init_ncoal_gamma_params(
        self,
        p=0.01,
        q=0.1,
        null_data=None,
        propintro=None,
    ):
        """Initialize the parameters for gamma mixture distribution."""
        accessible_index = np.where(self.mask == 1)[0]
        if null_data is None:
            null_data = self.ncoal[accessible_index]
        self.p = p
        self.q = q

        outliers, no_outlier_data = self.generalized_esd_test(
            data=null_data.flatten(), alpha=0.05, max_outlier_prop=0.1
        )
        self.emi2_mean1 = np.mean(no_outlier_data)
        self.emi2_var1 = np.var(no_outlier_data)
        self.emi2_b1 = self.emi2_mean1 / self.emi2_var1
        self.emi2_a1 = self.emi2_mean1**2 / self.emi2_var1

        subncoal = self.ncoal[accessible_index]
        # rescale emission so that density is between 0 and 1
        grid_max = gamma_sp.pdf(subncoal, self.emi2_a1, 1 / self.emi2_b1).max()
        self.scale_factor = 1 / grid_max

        if propintro is None:
            outliers = np.array(outliers)
            border = np.percentile(subncoal, 99)
            self.pi0 = 99 / 100
            if len(outliers) >= len(subncoal) * 0.01:
                border = np.min(outliers)
                self.pi0 = 1 - len(outliers) / len(subncoal)
        else:
            assert (propintro > 0) and (propintro < 50), (
                "proportion_admix should indicate admixture proportion from the minor ancestry"
                " between 0 and 50."
            )
            border = np.percentile(subncoal, 100 - propintro)
            self.pi0 = 1 - propintro / 100
        self.emi2_mean2 = np.mean(subncoal[subncoal >= border])
        self.emi2_var2 = np.var(subncoal[subncoal >= border])
        # handle extreme cases
        while self.emi2_var2 == 0:
            propintro = len(subncoal[subncoal >= border]) / len(subncoal)
            propintro = propintro + 0.01
            border = np.percentile(subncoal, 100 - propintro * 100)
            self.emi2_mean2 = np.mean(subncoal[subncoal >= border])
            self.emi2_var2 = np.var(subncoal[subncoal >= border])
            self.pi0 = 1 - propintro
        self.emi2_b2 = self.emi2_mean2 / self.emi2_var2
        zs1 = (subncoal >= border).astype("int")
        fa1 = lambda a: np.sum(
            zs1 * (np.log(subncoal) + np.log(self.emi2_b2) - digamma(a))
        )
        try:
            self.emi2_a2 = brentq(fa1, 1, 1e6)
        except ValueError:
            self.emi2_a2 = 1.1
        self.emi2_b2 = self.emi2_a2 * np.sum(zs1) / np.sum(subncoal * zs1)

    def forward_algo(self, p=1e-2, q=1e-2, emissions=None):
        """Implement the forward algorithm for the binary hmm."""
        assert (p > 0) and (q > 0)
        assert self.emissions is not None
        if emissions is None:
            emissions = self.emissions
        alphas, scaler, loglik = forward_algo_product(
            p=p,
            q=q,
            es=emissions,
            pi0=self.pi0,
        )
        return alphas, scaler, loglik

    def backward_algo(self, p=1e-2, q=1e-2, emissions=None):
        """Implement the backward algorithm for the binary hmm."""
        assert (p > 0) and (q > 0)
        assert self.emissions is not None
        if emissions is None:
            emissions = self.emissions
        betas, scaler, loglik = backward_algo_product(
            p=p,
            q=q,
            es=emissions,
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
        """Update for transition probabilities."""
        assert alphas.size == betas.size
        assert (p > 0) and (q > 0)
        assert (p < 1) and (q < 1)
        m = self.m
        eta_01 = np.zeros(m - 1)
        eta_10 = np.zeros(m - 1)
        eta_01, eta_10 = update_oneind_cython(
            alphas=alphas,
            betas=betas,
            emissions=self.emissions,
            p=p,
            q=q,
        )
        p_est = np.exp(eta_01).sum() / np.exp(gammas[0, :-1]).sum()
        q_est = np.exp(eta_10).sum() / np.exp(gammas[1, :-1]).sum()
        return p_est, q_est

    def baum_welch_ecm(
        self,
        niter=10,
        threshold=0.1,
        **kwargs,
    ):
        """Implement the Baum-welch algorithm with an ECM update step for the mixture of gammas distribution in the non-null case."""
        assert niter > 0

        loglik_acc = np.zeros(niter + 1)
        p_acc = np.zeros(niter + 1)
        q_acc = np.zeros(niter + 1)
        ea1_acc = np.zeros(niter + 1)
        eb1_acc = np.zeros(niter + 1)
        ea2_acc = np.zeros(niter + 1)
        eb2_acc = np.zeros(niter + 1)
        pi0_acc = np.zeros(niter + 1)
        iter_acc = np.zeros(niter + 1)
        p_acc[0] = self.p
        q_acc[0] = self.q
        ea2_acc[0] = self.emi2_a2
        eb2_acc[0] = self.emi2_b2
        ea1_acc[0] = self.emi2_a1
        eb1_acc[0] = self.emi2_b1
        pi0_acc[0] = self.pi0

        for i in tqdm(range(niter)):
            # update emissions z = 1
            iter_acc[i] = i
            self.emissions[1, :] = self.cache_emissions(z=1)
            # forward-backward algorithm
            gammas, alphas, betas, loglik = self.forward_backward_algo(
                p=self.p,
                q=self.q,
                emissions=self.emissions,
            )
            loglik_acc[i] = loglik
            # convergence check
            if i > 3:
                lk = loglik_acc[i - 1] - loglik_acc[i - 2]
                if lk < threshold:
                    break
            # update pi0
            pi0_hat = np.exp(gammas[0, 0])
            self.pi0 = pi0_hat

            # EM inference of emissions
            include_index = np.where(self.mask == 1)[0]
            fa1 = lambda a: np.sum(
                np.exp(gammas[1, include_index])
                * (
                    np.log(self.ncoal[include_index])
                    + np.log(self.emi2_b2)
                    - digamma(a)
                )
            )
            try:
                self.emi2_a2 = brentq(fa1, 1, 1e6)
            except ValueError:
                self.emi2_a2 = self.emi2_a2
            self.emi2_b2 = (
                self.emi2_a2
                * np.sum(np.exp(gammas[1, include_index]))
                / np.sum(self.ncoal[include_index] * np.exp(gammas[1, include_index]))
            )

            # EM inference of transitions
            p_est, q_est = self.update_transitions(
                np.copy(gammas[:, include_index], order="C"),
                np.copy(alphas[:, include_index], order="C"),
                np.copy(betas[:, include_index], order="C"),
                p=self.p,
                q=self.q,
            )
            if p_est >= 1e-5 and q_est >= 1e-5:
                self.p = p_est
                self.q = q_est

            pi0_acc[i + 1] = self.pi0
            p_acc[i + 1] = self.p
            q_acc[i + 1] = self.q
            ea2_acc[i + 1] = self.emi2_a2
            eb2_acc[i + 1] = self.emi2_b2
            ea1_acc[i + 1] = self.emi2_a1
            eb1_acc[i + 1] = self.emi2_b1

        res_dict = {
            "iters": iter_acc,
            "logliks": loglik_acc,
            "p": p_acc,
            "q": q_acc,
            "pi0": pi0_acc,
            "a1": ea1_acc,
            "b1": eb1_acc,
            "a2": ea2_acc,
            "b2": eb2_acc,
        }
        return res_dict

    def prepare_data_tmrca(self, ts, ind, subrange=None, t_archaic=20000):
        """Prepare data for HMM."""
        self.add_tree_sequence(ts, subrange=subrange)
        self.extract_ncoal(idx=ind, t_archaic=t_archaic, subrange=subrange)
        return self.ncoal, self.t1s, self.t2s, self.treespan, self.n_leaves

    def init_hmm(
        self,
        data,
        treespan,
        seed=42,
        intro_prop=None,
        subrange=None,
        include_regions=None,
        p=0.01,
        q=0.1,
    ):
        """Initialize HMM parameters using wrapper function."""
        np.random.seed(seed)
        self.treespan = treespan
        self.treespan_phy = np.copy(treespan, order="C")
        self.m = self.treespan.shape[0]
        self.p = p
        self.q = q
        self.set_constant_recomb()
        if include_regions is not None:
            self.mask = include_regions
        else:
            self.mask = np.ones(self.treespan.shape[0])
        self.emissions = np.zeros(shape=(2, self.m))
        self.ncoal = data
        if intro_prop is not None:
            intro_prop = intro_prop * 100
        self.init_ncoal_gamma_params(p=p, q=q, propintro=intro_prop)
        self.emissions[0] = self.cache_emissions(z=0)

    def train(self, niter=200, seed=42, threshold=0.1):
        """Train HMM using wrapper function."""
        np.random.seed(seed)
        res_dict = self.baum_welch_ecm(
            niter=niter,
            threshold=threshold,
        )
        return res_dict

    def decode(self, seed=42):
        """Decode HMM using wrapper function."""
        np.random.seed(seed)
        self.emissions[1] = self.cache_emissions(z=1)
        self.emissions[0] = self.cache_emissions(z=0)
        (
            gammas,
            alphas,
            betas,
            _,
        ) = self.forward_backward_algo(p=self.p, q=self.q, emissions=self.emissions)
        return gammas, alphas, betas
