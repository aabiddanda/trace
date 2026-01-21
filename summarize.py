import numpy as np
import tskit
import tszip
import sys
from arg_hmm.arg_hmm import GhostProductHmm
from arg_hmm.utils import (
    ExplicitDefaultsHelpFormatter,
    Output_utils
)
import argparse
import pandas as pd
import pybedtools

def main():
    parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--file",
        help="the posterior probability file from ARGHMM, end with .xss.npz. Multiple files are allowed, separated by comma",
        type=str,
        required=True,
    )
    required.add_argument(
        "--out",
        help='prefix for output file, output file will be named as [out].summary.txt',
        type=str,
        required=True,
    )
    optional = parser.add_argument_group("optional")
    optional.add_argument(
        "--chrom",
        help="chromosome ID used in the output file",
        type=str,
        default="chr1",
    )
    optional.add_argument(
        "--posterior-threshold",
        help="posterior probability threshold for calling introgression",
        type=float,
        default=0.9,
    )
    optional.add_argument(
        "--physical-length-threshold",
        help="physical length threshold for calling introgression, in bp",
        type=int,
        default=50000,
    )
    optional.add_argument(
        "--genetic-distance-threshold",
        help="genetic distance threshold for calling introgression, in cM",
        type=float,
        default=0.05,
    )
    optional.add_argument(
        "--remove-margin",
        help="remove margin from start and end in states, in kbp",
        type=float,
        default=0,
    )
    args = parser.parse_args()

    # read the args
    chrom = args.chrom
    pp_cutoff = args.posterior_threshold
    phy_cutoff = args.physical_length_threshold
    l_cutoff = args.genetic_distance_threshold
    out_prefix = args.out
    file = args.file
    remove_margin = int(args.remove_margin) * 1000  # convert kbp to bp

    # read the posterior probability file
    # NEED: combine different chroms
    files = file.split(",")
    try:
        with np.load(files[0]) as data:
            treespan = data["treespan"]
            treespan_phy = data["treespan_phy"]
            window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
        pp = np.zeros(shape = (len(files), 2, len(treespan)))
        for i, file in enumerate(files):
            with np.load(file) as d:
                data = {k: d[k] for k in d.files}
                pp[i] = data["gammas"]
        pp = np.mean(pp, axis=0)
    except Exception as e:
        print(f"Error reading the posterior probability file: {e}")
        print(files)
        sys.exit(1)

    states = Output_utils().summarize(
        pp = pp,
        treespan = treespan,
        treespan_phy = treespan_phy,
        outpref = out_prefix,
        chrom = chrom,
        pp_cutoff = pp_cutoff,
        phy_cutoff = phy_cutoff,
        l_cutoff = l_cutoff,
        remove_margin = int(remove_margin / window_size)  # convert bp to number of windows,
    )
    if len(files) == 1:
        data["states"] = states
        np.savez_compressed(files[0], **data)

if __name__ == "__main__":
    main()   