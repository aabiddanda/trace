"""CLI for trace-extract."""
import logging
import shutil
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tskit
import tszip
from tqdm import tqdm

from tracehmm import TRACE, OutputUtils

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def verify_indivs(indiv, sample_names=None):
    """Verify the structure of the individuals provided for extraction."""
    indiv = indiv.strip("\n").strip(",").split(",")
    assert len(indiv) > 0, "No individuals provided ... exiting."
    if sample_names is not None:
        assert Path(
            sample_names
        ).is_file(), f"Sample names file {sample_names} does not exist ... exiting."
    try:
        indiv = [int(x) for x in indiv if len(x) > 0]
    except ValueError:
        indiv = [str(x) for x in indiv if len(x) > 0]
    output_utils = OutputUtils(samplefile=sample_names, samplename=indiv)
    if sample_names is not None:
        samplename_to_tsid, tsid_to_samplename = output_utils.read_samplename()
        if isinstance(indiv[0], str):
            for x in indiv:
                if x not in samplename_to_tsid:
                    logging.info(
                        f"Sample name {x} not found in sample names file ... exiting."
                    )
                    sys.exit(1)
            indiv = np.array([samplename_to_tsid[x] for x in indiv])
    assert isinstance(indiv[0], int)
    return indiv


def get_data(ts, ind, t_archaic, windowsize, mask=None, chrom=None):
    """Extract data from tree sequence for a specific individual haplotype."""
    hmm = TRACE()
    tncoal, tt1s, tt2s, treespan, tnleaves = hmm.prepare_data_tmrca(
        ts=ts, ind=ind, t_archaic=t_archaic
    )
    if mask is not None:
        mask = hmm.mask_regions(treespan, chrom, mask, f=0.99)
    else:
        mask = np.ones(treespan.shape[0])
    treespan = treespan.astype(int)
    if windowsize is None:
        return tncoal, tt1s, tt2s, tnleaves, treespan, mask, mask
    genome_length = ts.sequence_length
    m = int(genome_length / windowsize) + int(genome_length % windowsize > 0)
    ncoal_sub = np.zeros((len(ind), m))
    t1s_sub = np.zeros((len(ind), m))
    t2s_sub = np.zeros((len(ind), m))
    nleaves_sub = np.zeros((len(ind), m))
    accessible_windows = np.ones(m)
    if len(ind) == 1:
        tncoal = np.array([tncoal])
        tt1s = np.array([tt1s])
        tt2s = np.array([tt2s])
        tnleaves = np.array([tnleaves])
    t = 0
    curtrees = []
    for k in tqdm(range(m)):
        while t < treespan.shape[0] and treespan[t][0] < int(
            k * windowsize + windowsize
        ):
            if mask[t] == 1:
                curtrees.append(t)
            else:
                curtrees.append(-1)
            t += 1
        if len(curtrees) == 0:
            for i in range(len(ind)):
                ncoal_sub[i][k] = tncoal[i][t - 1]
                t1s_sub[i][k] = tt1s[i][t - 1]
                t2s_sub[i][k] = tt2s[i][t - 1]
                nleaves_sub[i][k] = tnleaves[i][t - 1]
        else:
            treelens = []
            curtrees = np.array(curtrees)
            curtrees = curtrees[curtrees >= 0]
            if len(curtrees) == 0:
                accessible_windows[k] = 0
                if k == 0:
                    for i in range(len(ind)):
                        ncoal_sub[i][k] = 1e-10
                        t1s_sub[i][k] = 0
                        t2s_sub[i][k] = 0
                        nleaves_sub[i][k] = 0
                else:
                    for i in range(len(ind)):
                        ncoal_sub[i][k] = 1e-10
                        t1s_sub[i][k] = 0
                        t2s_sub[i][k] = 0
                        nleaves_sub[i][k] = 0
            else:
                for j in range(len(curtrees)):
                    treelens.append(
                        min(treespan[curtrees[j]][1], int(k * windowsize + windowsize))
                        - max(treespan[curtrees[j]][0], int(k * windowsize))
                    )
                treelens = np.array(treelens)
                curtrees = curtrees[treelens > 1]
                treelens = treelens[treelens > 1]
                if len(curtrees) == 0:
                    accessible_windows[k] = 0
                    for i in range(len(ind)):
                        ncoal_sub[i][k] = 1e-10
                        t1s_sub[i][k] = 0
                        t2s_sub[i][k] = 0
                        nleaves_sub[i][k] = 0
                else:
                    for i in range(len(ind)):
                        ncoal_sub[i][k] = np.average(
                            tncoal[i][curtrees], weights=treelens
                        )
                        t1s_sub[i][k] = np.average(tt1s[i][curtrees], weights=treelens)
                        t2s_sub[i][k] = np.average(tt2s[i][curtrees], weights=treelens)
                        nleaves_sub[i][k] = np.average(
                            tnleaves[i][curtrees], weights=treelens
                        )
            curtrees = []
            if treespan[t - 1][1] < (k + 1) * windowsize + windowsize:
                if mask[t - 1] == 1:
                    curtrees.append(t - 1)
                else:
                    curtrees.append(-1)
    return ncoal_sub, t1s_sub, t2s_sub, nleaves_sub, treespan, accessible_windows, mask


@click.command(context_settings={"show_default": True})
@click.option(
    "--tree-file",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="Input data in tskit or tsz format.",
)
@click.option(
    "--t-archaic",
    "-t",
    required=True,
    type=float,
    help="Focal time for branch.",
)
@click.option(
    "--individuals",
    "-i",
    required=True,
    type=str,
    help="List of sampled haplotypes to run analysis for, comma separated (no spaces). "
    + "Recognizes tree node IDs (int).",
    default=None,
)
@click.option(
    "--window-size",
    "-w",
    required=False,
    type=int,
    default=None,
    help="Window size summarizing tree sequences (required if working with multiple posterior tree sequences "
    + "like outputs from SINGER). If not provided, uses the marginal trees directly.",
)
@click.option(
    "--chrom",
    help="chromosome ID for the tree sequence, must match the chromosome ID in the include regions file.",
    type=str,
    default=None,
)
@click.option(
    "--include-regions",
    help="A BED file containing the INCLUDE regions for the tree sequence.",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="trace",
    help="Output file prefixes.",
)
def main(
    tree_file=None,
    t_archaic=15e3,
    individuals=None,
    window_size=None,
    chrom=None,
    include_regions=None,
    out="trace",
):
    """TRACE-Extract CLI."""
    logging.info("Starting trace-extract ...")
    logging.info(f"Loading {tree_file} ... ")
    if tree_file.endswith(".trees"):
        ts = tskit.load(tree_file)
    elif tree_file.endswith(".tsz"):
        ts = tszip.decompress(tree_file)
    else:
        logging.info(f"Unrecognized file extension: {tree_file}! exiting ...")
        sys.exit(1)

    # NOTE: you probably have to error out to make sure both are not None...
    logging.info("Verifying individual labels ...")
    sample_names = None  # Currently disabled
    indiv = verify_indivs(individuals, sample_names)
    if include_regions is not None:
        if shutil.which("bedtools") is None:
            raise ValueError(
                "No detectable `bedtools` installation for `--include-regions`. Please install for your system!"
            )
            sys.exit(1)
        if chrom is None:
            logging.info(
                "chromosome identifier is not specified (required when using --include-regions) ... exiting."
            )
            sys.exit(1)
        # check include regions file is a valid bed file
        include_regions_df = pd.read_csv(
            include_regions, sep="\t", header=None, names=["chrom", "start", "end"]
        )
        assert (
            len(include_regions_df.columns) == 3
        ), "Include regions file must be a valid BED file with 3 columns: chrom, start, end."
        assert (
            len(include_regions_df["chrom"].unique()) == 1
        ), "Include regions file must be a valide BED file (no header) and contain only one chromosome."
        assert include_regions_df["chrom"].unique()[0] == chrom, (
            "Chromosome ID in include regions file must match the provided --chrom argument.\n"
            f"Provided chromosome ID: {chrom}, chromosome ID in include regions file: {include_regions_df['chrom'].unique()[0]}"
        )
    logging.info(
        f"Extracting TRACE-information from {tree_file} across {len(indiv)} individuals ..."
    )
    ncoal, t1s, t2s, nleaves, treespan, accessible_windows, mask = get_data(
        ts, indiv, t_archaic, window_size, include_regions, chrom
    )
    if window_size is not None:
        m = int(ts.sequence_length / window_size) + int(
            ts.sequence_length % window_size > 0
        )
        atreespan = np.array(
            [[t * window_size, (t + 1) * window_size] for t in range(m)]
        )
    else:
        atreespan = treespan

    logging.info(f"Writing output to {out}.npz ...")
    np.savez_compressed(
        f"{out}.npz",
        ncoal=ncoal,
        t1s=t1s,
        t2s=t2s,
        marginal_treespan=treespan,
        treespan=atreespan,
        marginal_mask=mask,
        accessible_windows=accessible_windows,
        individuals=indiv,
    )
