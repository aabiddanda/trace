"""CLI for TRACE."""
import logging
import pathlib
import sys

import click
import numpy as np
import pandas as pd

from tracehmm import TRACE, OutputUtils

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command(context_settings={"show_default": True})
@click.option(
    "--files",
    help="Posterior probability file from trace-infer, end with .xss.npz. Multiple files (for the same individual, "
    + "different chromosomes) are allowed, separated by comma",
    type=str,
    required=True,
)
@click.option(
    "--chroms",
    help="Chromosome ID used in the output file, must be consistent with the input files",
    type=str,
    required=True,
)
@click.option(
    "--posterior-threshold",
    "-p",
    help="posterior probability threshold for calling introgression",
    type=float,
    default=0.9,
)
@click.option(
    "--physical-length-threshold",
    help="physical length threshold for calling introgression, in bp",
    type=int,
    default=50000,
)
@click.option(
    "--genetic-distance-threshold",
    help="genetic distance threshold for calling introgression, in cM",
    type=float,
    default=0.05,
)
@click.option(
    "--out",
    help="prefix for output file, output file will be named as [out].summary.txt",
    type=str,
    required=True,
)
def main(
    files,
    chroms,
    posterior_threshold,
    physical_length_threshold,
    genetic_distance_threshold,
    out="trace",
):
    """TRACE-Summarize CLI."""
    logging.info(f"Starting TRACE-summarize ...")

    # read the posterior probability file
    files = files.split(",")
    chroms = chroms.split(",")
    assert len(chroms) == len(
        files
    ), "Number of chromosomes must match number of files!"
    out_pps = []
    out_treespans = []
    out_treespans_phy = []
    try:
        for file in files:
            with np.load(file) as d:
                data = {k: d[k] for k in d.files}
                out_pps.append(data["gammas"])
                out_treespans.append(data["treespan"])
                out_treespans_phy.append(data["treespan_phy"])
    except Exception as e:
        logging.info(f"Error reading the posterior probability file: {e}")
        logging.info(f"Total files: {files}")
        sys.exit(1)

    logging.info(f"Writing output to {out}.summary.txt ...")
    outstring = "chromosome\tstart\tend\tmean_posterior\tlength(bp)\tlength(cM)\n"
    for i in range(len(out_pps)):
        pp = out_pps[i]
        treespan = out_treespans[i]
        treespan_phy = out_treespans_phy[i]
        chrom = chroms[i]
        out_chunk, states = OutputUtils().summarize(
            pp=pp,
            treespan=treespan,
            treespan_phy=treespan_phy,
            outpref=out,
            chrom=chrom,
            pp_cutoff=posterior_threshold,
            phy_cutoff=physical_length_threshold,
            l_cutoff=genetic_distance_threshold,
        )
        outstring += out_chunk
        with np.load(files[i]) as d:
            data = {k: d[k] for k in d.files}
            data["states"] = states
        np.savez_compressed(files[i], **data)
    outfile = open(str(out) + ".summary.txt", "w")
    outfile.write(outstring)
    outfile.close()
    logging.info(f"Output written to {out}.summary.txt!")
