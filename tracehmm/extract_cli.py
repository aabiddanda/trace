"""CLI for trace-extract."""
import logging
import sys
from tracehmm import TRACE

import click
import numpy as np
import pandas as pd

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_data(ts, ind, t_archaic, windowsize, func, mask=None, chrom=None):
    genome_length = ts.sequence_length
    m = int(genome_length / windowsize) + int(genome_length % windowsize > 0)
    ncoal_sub = np.zeros((len(ind), m))
    t1s_sub = np.zeros((len(ind), m))
    t2s_sub = np.zeros((len(ind), m))
    nleaves_sub = np.zeros((len(ind), m))
    hmm = TRACE()
    tncoal, tt1s, tt2s, treespan, tnleaves = hmm.prepare_data_tmrca(
        ts=ts, ind=ind, t_archaic=t_archaic
    )
    if mask is not None:
        mask = hmm.mask_regions(treespan, chrom, mask, f=0.99)
    else:
        mask = np.ones(treespan.shape[0])
    accessible_windows = np.ones(m)
    treespan = treespan.astype(int)
    if len(ind) == 1:
        tncoal = np.array([tncoal])
        tt1s = np.array([tt1s])
        tt2s = np.array([tt2s])
        tnleaves = np.array([tnleaves])
    t = 0
    curtrees = []
    for k in range(m):
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


@click.command()
@click.option(
    "--tree-file",
    required=True,
    type=click.Path(exists=True),
    help="Input data in tskit or tsz format.",
)
@click.option(
    "--t-archaic",
    required=True,
    type=float,
    default=15e3,
    help="Focal time for branch.",
)
@click.option(
    "--samples",
    "-s",
    required=False,
    type=click.Path(exists=True),
    help="List of sampled individuals to run analysis for.",
)
@click.option(
    "--window-size",
    "-w",
    required=False,
    type=int,
    default=1000,
    help="Window size summarizing SINGER tree sequences.",
)
@click.option(
    "--func",
    "-f",
    required=False,
    type=click.Choice(["mean", "median"]),
    default="mean",
    help="Summarize function for windows.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="trace",
    help="Output file prefix.",
)
def main(
    input=None,
    time=15e3,
    out="trace",
):
    """TRACE-Extract CLI."""
    logging.info(f"Starting trace-extract ...")
    print(f"loading {input}")
    if tree_file.endswith(".trees"):
        ts = tskit.load(tree_file)
    elif tree_file.endswith(".tsz"):
        ts = tszip.decompress(tree_file)
    else:
        print(f"Unrecognized file extension: {tree_file}")
        sys.exit(1)
