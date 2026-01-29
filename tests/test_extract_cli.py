"""Test suite for Extract CLI."""

import os
from pathlib import Path

import msprime
import numpy as np
import pytest
import tszip


@pytest.fixture(scope="session")
def tszip1(tmp_path_factory):
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
    return fn


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
    fn = tmp_path_factory.mktemp("extract_data") / "chr1_regions.txt"
    with open(fn, "w+") as out:
        out.write("chrX\t0\t1000000\tX\n")
        out.write("chrX\t2000000\t1500000\tY\n")
    return fn


def check_npz_file(fp):
    """Check an extracted npz file."""
    assert Path(fp).is_file()
    data = np.load(fp)
    keys = [
        "ncoal",
        "t1s",
        "t2s",
        "treespan",
        "marginal_treespan",
        "marginal_mask",
        "accessible_windows",
        "individuals",
    ]
    for k in data.keys():
        assert k in keys


def test_extract_ts1(tszip1, tmp_path_factory):
    """Test extraction of a standard tszip file."""
    out_fp = Path(tszip1).with_suffix(".npz")
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --individuals 0,1,2 --out {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()
    check_npz_file(out_fp)
    data = np.load(out_fp)
    assert data["individuals"].size == 3


def test_check_bad_ts(dumb_file):
    """Test extraction with poor file input."""
    outfix = Path(dumb_file).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {dumb_file} --individuals 0 --out {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("indiv", ["200", "A,B,C", "-100"])
def test_bad_indivs(tszip1, indiv):
    """Test specifying bad individuals."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --individuals {indiv} --out {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("t", [-1, 1e20])
def test_t_archaic(tszip1, t):
    """Test different estimates of t-archaic."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --t-archaic {t} --out {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("w", [100, 1000, 10000, 100000])
def test_window_size(tszip1, w):
    """Test different window sizes."""
    out_fp = Path(tszip1).with_suffix(".npz")
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --individuals 0,1,2 --window-size {w} --out {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()


@pytest.mark.parametrize("w", [None, 0, -100, 100.0])
def test_bad_window_size(tszip1, w):
    """Test extraction with bad window sizes."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --individuals 0,1,2 --window-size {w} --out {outfix}"
    )
    assert exit_status != 0


def test_chrom_regions(tszip1, proper_chr1_regions):
    """Test defining chromosomal regions."""
    out_fp = Path(tszip1).with_suffix(".npz")
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --individuals 0,1,2 --chrom chr1 --include-regions {proper_chr1_regions} --out {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()
    check_npz_file(out_fp)


@pytest.mark.parametrize("c", [None, "chr2", "chrX", "X"])
def test_chrom_mismatch(tszip1, proper_chr1_regions, c):
    """Test that chromosome mismatch is not supported."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --individuals 0,1,2 --chrom {c} --include_regions {proper_chr1_regions} --out {outfix}"
    )
    assert exit_status != 0

def test_chrom_regions(tszip1, bad_chr_regions):
    """Test defining chromosomal regions."""
    out_fp = Path(tszip1).with_suffix(".npz")
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --individuals 0,1,2 --chrom chr1 --include-regions {bad_chr_regions} --out {outfix}"
    )
    assert exit_status != 0
