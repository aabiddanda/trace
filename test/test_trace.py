"""Testing suite for TRACE HMM functions."""

import msprime as msp
import pytest

from tracehmm import TRACE

ts1 = msp.sim_ancestry(
    samples=100,
    population_size=1e4,
    recombination_rate=1e-8,
    random_seed=42,
    sequence_length=5e6,
)
ts2 = msp.sim_ancestry(
    samples=50,
    population_size=1e4,
    recombination_rate=5e-8,
    random_seed=24,
    sequence_length=5e6,
)


def test_init():
    """Test that trace can be naively initialized."""
    hmm = TRACE()
    assert hmm is not None


@pytest.mark.parametrize("ts", [ts1, ts2])
def test_add_ts(ts):
    """Test that adding a tree-sequence works fine."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    assert hmm.ts is not None
    assert hmm.ts.num_samples > 0


@pytest.mark.parametrize("ts", [ts1, ts2])
def test_extract_ncoal(ts):
    """Test extraction of number of coalescent events from TRACE."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    # NOTE: there is some funkiness about the random seed setting here ...
    ncoal, t1s, t2s, n_leaves = hmm.extract_ncoal(idx=0, t_archaic=15e3)
    assert ncoal.size > 0


@pytest.mark.parametrize("ts", [ts1, ts2])
def test_extract_ncoal_bad_idx(ts):
    """Test extraction of number of coalescent events from TRACE."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    # NOTE: there is some funkiness about the random seed setting here ...
    assert hmm.ts is not None
    x = int(2 * hmm.ts.num_samples + 1)
    with pytest.raises(ValueError):
        ncoal, t1s, t2s, n_leaves = hmm.extract_ncoal(idx=x, t_archaic=15e3)


@pytest.mark.parametrize("ts", [ts1, ts2])
@pytest.mark.parametrize("t", [-100, 0.0, "x"])
def test_extract_ncoal_bad_time(ts, t):
    """Test extraction of number of coalescent events from TRACE."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    # NOTE: there is some funkiness about the random seed setting here ...
    assert hmm.ts is not None
    with pytest.raises(AssertionError, TypeError):
        ncoal, t1s, t2s, n_leaves = hmm.extract_ncoal(idx=0, t_archaic=t)
