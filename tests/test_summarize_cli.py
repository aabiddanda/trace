"""Test suite for Summarize CLI."""

import os
from pathlib import Path

import msprime
import numpy as np
import pandas as pd
import pytest
import tszip


@pytest.fixture(scope="session")
def ts_infer_chr1(tmp_path_factory):
    """Create temporary tszip file."""
    ts2 = msprime.sim_ancestry(
        samples=10,
        population_size=1e4,
        recombination_rate=1e-8,
        random_seed=42,
        sequence_length=10e6,
    )
    fn = tmp_path_factory.mktemp("extract_data") / "ts2.tsz"
    tszip.compress(ts2, fn)
    out_fp = Path(fn).with_suffix(".npz")
    out_fp2 = Path(fn).with_suffix(".chr1.xss.npz")
    outfix = Path(fn).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {fn} -t 15e3 --individuals 0,1,2 --chrom chr1 --out {outfix}"
    )
    assert exit_status == 0
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {out_fp} --chroms chr1 -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp2).is_file()
    return out_fp2


@pytest.fixture(scope="session")
def ts_infer_chr2(tmp_path_factory):
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
    out_fp2 = Path(fn).with_suffix(".chr2.xss.npz")
    outfix = Path(fn).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {fn} -t 15e3 --individuals 0,1,2 --chrom chr2 --out {outfix}"
    )
    assert exit_status == 0
    exit_status = os.system(
        f"trace-infer -i 2 --npz-files {out_fp} --chroms chr2 -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp2).is_file()
    return out_fp2


def eval_summary_file(fp, chroms=["chr1"]):
    """Perform a small audit on the output file."""
    df = pd.read_csv(fp, sep="\t")
    for x in df.columns:
        assert x in [
            "chromosome",
            "start",
            "end",
            "mean_posterior",
            "length(bp)",
            "length(cM)",
        ]
    assert df["chromosome"].unique().size <= len(chroms)
    for s, e in zip(df["start"], df["end"]):
        assert e >= s


def test_simple_summarize(ts_infer_chr1):
    """Test extraction of a standard tszip file."""
    out_fp = Path(ts_infer_chr1).with_suffix(".summary.txt")
    outfix = Path(ts_infer_chr1).with_suffix("")
    exit_status = os.system(
        f"trace-summarize -f {ts_infer_chr1} --chroms chr1 -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()
    eval_summary_file(out_fp)


def test_multichrom_summarize(ts_infer_chr1, ts_infer_chr2):
    """Test extraction of a standard tszip file."""
    out_fp = Path(ts_infer_chr1).with_suffix(".summary.txt")
    outfix = Path(ts_infer_chr1).with_suffix("")
    exit_status = os.system(
        f"trace-summarize -f {ts_infer_chr1},{ts_infer_chr2} --chroms chr1,chr2 -o {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()
    eval_summary_file(out_fp, chroms=["chr1", "chr2"])


@pytest.mark.parametrize("ppthresh", [0.0, 1.1, None, "X"])
def test_bad_ppthresh(ts_infer_chr1, ppthresh):
    """Test extraction of a standard tszip file."""
    outfix = Path(ts_infer_chr1).with_suffix("")
    exit_status = os.system(
        f"trace-summarize -f {ts_infer_chr1} --posterior-threshold {ppthresh} --chroms chr1 -o {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("gendist", [0.0, -1, None, "X"])
def test_bad_gendist(ts_infer_chr1, gendist):
    """Test extraction of a standard tszip file."""
    outfix = Path(ts_infer_chr1).with_suffix("")
    exit_status = os.system(
        f"trace-summarize -f {ts_infer_chr1} --genetic-distance-threshold {gendist} --chroms chr1 -o {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("bpdist", [0, -1, None, "X"])
def test_bad_physdist(ts_infer_chr1, bpdist):
    """Test extraction of a standard tszip file."""
    outfix = Path(ts_infer_chr1).with_suffix("")
    exit_status = os.system(
        f"trace-summarize -f {ts_infer_chr1}  --physical-length-threshold {bpdist} --chroms chr1 -o {outfix}"
    )
    assert exit_status != 0
