"""Test suite for Extract CLI."""

import os
from pathlib import Path

import msprime
import numpy as np
import pytest
import tszip


@pytest.fixture(scope="session")
def ts_extract1(tmp_path_factory):
    """Create temporary tszip file."""
    ts1 = msprime.sim_ancestry(
        samples=10,
        population_size=1e4,
        recombination_rate=1e-8,
        random_seed=42,
        sequence_length=5e6,
    )
    fn = tmp_path_factory.mktemp("extract_data") / "ts1.tsz"
    tszip.compress(ts1, fn)
    out_fp = Path(fn).with_suffix(".npz")
    outfix = Path(fn).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {fn} --individuals 0,1,2 --out {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()
    data = np.load(out_fp)
    assert data["individuals"].size == 3
    return out_fp


@pytest.fixture(scope="session")
def ts_extract2(tmp_path_factory):
    """Create temporary tszip file."""
    ts2 = msprime.sim_ancestry(
        samples=10,
        population_size=1e4,
        recombination_rate=1e-8,
        random_seed=24,
        sequence_length=5e6,
    )
    fn = tmp_path_factory.mktemp("extract_data") / "ts2.tsz"
    tszip.compress(ts2, fn)
    out_fp = Path(fn).with_suffix(".npz")
    outfix = Path(fn).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {fn} --individuals 0,1,2 --out {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()
    data = np.load(out_fp)
    assert data["individuals"].size == 3
    return out_fp


@pytest.fixture(scope="session")
def dumb_file(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "xxx.txt"
    with open(fn, "w+") as out:
        out.write("Hello\nWorld\n")
    return fn


@pytest.fixture(scope="session")
def proper_chr1_regions(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "chr1_regions.txt"
    with open(fn, "w+") as out:
        out.write("chr1\t0\t1000000\n")
        out.write("chr1\t2000000\t4000000\n")
    return fn


@pytest.fixture(scope="session")
def bad_chr_regions(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "chr1_regions_bad.txt"
    with open(fn, "w+") as out:
        out.write("chrX\t0\t1000000\tX\n")
        out.write("chrX\t2000000\t1500000\tY\n")
    return fn


def check_xss_npz_file(fp):
    """Check an extracted npz file."""
    assert Path(fp).is_file()
    data = np.load(fp)
    keys = [
        "ncoal",
        "treespan",
        "treespan_phy",
        "accessible_windows",
        "params",
        "gammas",
        "seed",
        "individual",
    ]
    for k in data.keys():
        assert k in keys


def test_simple_infer(ts_extract1):
    """Test extraction of a standard tszip file."""
    out_fp = Path(ts_extract1).with_suffix(".chr1.xss.npz")
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(f"trace-infer -i 2 --npz-files {ts_extract1} -o {outfix}")
    assert exit_status == 0
    assert Path(out_fp).is_file()
    check_xss_npz_file(out_fp)


def test_two_chroms(ts_extract1, ts_extract2):
    """Test extraction of a standard tszip file."""
    out_fp1 = Path(ts_extract1).with_suffix(".chr1.xss.npz")
    out_fp2 = Path(ts_extract1).with_suffix(".chr2.xss.npz")
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {ts_extract1},{ts_extract2} --chroms chr1,chr2 -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp1).is_file()
    assert Path(out_fp2).is_file()
    check_xss_npz_file(out_fp1)
    check_xss_npz_file(out_fp2)
