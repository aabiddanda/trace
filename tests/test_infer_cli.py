"""Test suite for Infer CLI."""

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
def genetic_map(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "genmap.txt"
    with open(fn, "w+") as out:
        out.write("chr1\t0\t0.01\t0.0\n")
        out.write("chr1\t1000000\t0.01\t1.0\n")
        out.write("chr1\t2000000\t0.01\t2.0\n")
        out.write("chr1\t3000000\t0.01\t3.0\n")
        out.write("chr1\t4000000\t0.01\t4.0\n")
    return fn


@pytest.fixture(scope="session")
def nonmono_genetic_map(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "genmap_bad.txt"
    with open(fn, "w+") as out:
        out.write("chr1\t0\t0.01\t0.0\n")
        out.write("chr1\t1000000\t0.01\t1.0\n")
        out.write("chr1\t2000000\t0.01\t0.8\n")
        out.write("chr1\t3000000\t0.01\t3.0\n")
        out.write("chr1\t4000000\t0.01\t4.0\n")
    return fn


@pytest.fixture(scope="session")
def chr2_genetic_map(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "genmap_chr2.txt"
    with open(fn, "w+") as out:
        out.write("chr2\t0\t0.01\t0.0\n")
        out.write("chr2\t1000000\t0.01\t1.0\n")
        out.write("chr2\t2000000\t0.01\t2.0\n")
        out.write("chr2\t3000000\t0.01\t3.0\n")
        out.write("chr2\t4000000\t0.01\t4.0\n")
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


@pytest.mark.parametrize("c", ["chr1", "", "X", "chr1,chr2,chr3"])
def test_bad_chroms(ts_extract1, ts_extract2, c):
    """Test extraction of a standard tszip file."""
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {ts_extract1},{ts_extract2} --chroms {c} -o {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("f", [None, "", "meanx", "range"])
def test_bad_func(ts_extract1, f):
    """Test extraction of a standard tszip file."""
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {ts_extract1},{ts_extract2} --func {f} -o {outfix}"
    )
    assert exit_status != 0


def test_genmap(ts_extract1, genetic_map):
    """Test extraction of a standard tszip file."""
    out_fp1 = Path(ts_extract1).with_suffix(".chr1.xss.npz")
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {ts_extract1} --genetic-maps {genetic_map} -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp1).is_file()
    check_xss_npz_file(out_fp1)


def test_multiple_genmap(ts_extract1, ts_extract2, genetic_map, chr2_genetic_map):
    """Test extraction of a standard tszip file."""
    out_fp1 = Path(ts_extract1).with_suffix(".chr1.xss.npz")
    out_fp2 = Path(ts_extract1).with_suffix(".chr2.xss.npz")
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {ts_extract1},{ts_extract2} --chroms chr1,chr2 --genetic-maps {genetic_map},{chr2_genetic_map} -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp1).is_file()
    assert Path(out_fp2).is_file()
    check_xss_npz_file(out_fp1)
    check_xss_npz_file(out_fp2)


@pytest.mark.parametrize("bad_map", [nonmono_genetic_map, dumb_file])
def test_bad_genmap(ts_extract1, bad_map):
    """Test extraction of a standard tszip file."""
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {ts_extract1} --chroms chr1 --genetic-maps {bad_map} -o {outfix}"
    )
    assert exit_status != 0


def test_datafiles_multichrom(ts_extract1, ts_extract2, tmp_path_factory):
    """Test the datafiles flag with multiple chromosomes."""
    out_fp1 = Path(ts_extract1).with_suffix(".chr1.xss.npz")
    out_fp2 = Path(ts_extract1).with_suffix(".chr2.xss.npz")
    # Create a temporary datafile
    chr1_data = tmp_path_factory.mktemp("extract_data") / "chr1_data.txt"
    chr2_data = tmp_path_factory.mktemp("extract_data") / "chr2_data.txt"
    with open(chr1_data, "w+") as fp:
        fp.write(str(Path(ts_extract1).resolve()) + "\n")
    with open(chr2_data, "w+") as fp:
        fp.write(str(Path(ts_extract2).resolve()) + "\n")
    assert Path(chr1_data).is_file()
    assert Path(chr2_data).is_file()
    outfix = Path(ts_extract1).with_suffix("")
    exit_status = os.system(
        f"trace-infer -i 2 --data-files {chr1_data},{chr2_data} --chroms chr1,chr2 -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp1).is_file()
    assert Path(out_fp2).is_file()
    check_xss_npz_file(out_fp1)
    check_xss_npz_file(out_fp2)
