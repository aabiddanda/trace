"""CLI for trace-extract."""
import logging
import sys

import click
import numpy as np
import pybedtools
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


def write_mutation_ages(ts, chrom=None, include_regions=None, outfix="trace"):
    """Write out the mutational ages."""
    out = ""
    for tree in tqdm(ts.trees()):
        for mut in tree.mutations():
            if tree.parent(mut.node) != tskit.NULL:
                out += f"{chrom}\t{int(ts.site(mut.site).position) - 1}\t{int(ts.site(mut.site).position)}\t{tree.time(mut.node)}_{tree.time(tree.parent(mut.node))}\n"
            else:
                out += f"{chrom}\t{int(ts.site(mut.site).position) - 1}\t{int(ts.site(mut.site).position)}\t{tree.time(mut.node)}_{tree.time(mut.node)}\n"
    a = pybedtools.BedTool(out, from_string=True)
    if include_regions is not None:
        logging.info(f"Loading {include_regions} to subset mutations considered ...")
        include_regions = pybedtools.BedTool(include_regions)
        a = a.intersect(include_regions, u=True)
    out = "chromosome\tposition\tmutation_age\n"
    for x in a:
        out += f"{x.chrom}\t{x.end}\t{x[3]}\n"
    with open(f"{outfix}.mutation_ages.txt", "w") as out_fp:
        out_fp.write(out)
    logging.info(f"Mutation ages saved to {outfix}.mutation_ages.txt!")


def verify_indivs(indiv=None, sample_names=None):
    """Verify the structure of the individuals provided for extraction."""
    if (indiv is None) and (sample_names is None):
        logging.info("Need to supply either --samples or --sample-names ... exiting.")
        sys.exit(1)
    if indiv is not None:
        indiv = indiv.strip("\"'").strip(",").split(",")
    else:
        try:
            with open(sample_names, "r") as f:
                indiv = f.readlines()
            indiv = [x.strip() for x in indiv]
        except FileNotFoundError:
            logging.info(f"{sample_names} is not a valid filepath...")
            sys.exit(1)
    try:
        indiv = [int(x) for x in indiv if len(x) > 0]
    except Exception as _:
        indiv = [str(x) for x in indiv if len(x) > 0]
    output_utils = OutputUtils(samplefile=sample_names, samplename=indiv)
    if sample_names is not None:
        samplename_to_tsid, tsid_to_samplename = output_utils.read_samplename()
        if isinstance(indiv[0], str):
            indiv = np.array([samplename_to_tsid[x] for x in indiv])

    # makesure indiv is tree node ID
    assert isinstance(indiv[0], int)
    return indiv


def get_data(ts, ind, t_archaic, windowsize, mask=None, chrom=None):
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
    type=str,
    help="List of sampled individuals to run analysis for, comma separated (no spaces). "
    + "Recognizes tree node IDs (int, default) or sample names (str, if --sample-names is provided).",
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
    "--sample-names",
    help="a file containing sample names for all individuals in the tree sequence, "
    + "tab separated, two columns, first column contains tree node id (int), "
    + "second column contains sample names (str)",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--chrom",
    help="chromosome ID for the tree sequence, must match the chromosome ID in the include regions file",
    type=str,
    default=None,
)
@click.option(
    "--include-regions",
    help="a BED file containing the INCLUDE regions for the tree sequence",
    type=click.Path(exists=True),
    default=None,
)
# @click.option(
#     "--mutation-age",
#     help="only extract mutation ages in the tree sequence, limited by include regions if specified",
#     is_flag=True,
#     default=False,
# )
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
    samples=None,
    window_size=None,
    sample_names=None,
    chrom=None,
    include_regions=None,
    # mutation_age=None,
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

    # if mutation_age:
    #     logging.info(f"Extracting mutations to write from {tree_file} ...")
    #     write_mutation_ages(
    #         ts, chrom=chrom, include_regions=include_regions, outfix=out
    #     )
    # else:
    # NOTE: you probably have to error out to make sure both are not None...
    logging.info("Verifying individual labels ...")
    logging.info(f"Comparing {samples} and {sample_names} ...")
    indiv = verify_indivs(samples, sample_names)
    if include_regions is not None:
        if chrom is None:
            logging.info(
                "chromosome identifier is not specified (required when using --include-regions) ... exiting."
            )
            sys.exit(1)
    # This is the actual look to run ...
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
    logging.info(
        f"Extracting TRACE-information from {tree_file} across {len(indiv)} individuals ..."
    )

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
