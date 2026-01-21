import tskit
import tszip
import sys
from trace.trace import TRACE
import argparse
import pandas as pd
from trace.utils import (
    ExplicitDefaultsHelpFormatter,
    Output_utils
)
import numpy as np
import pybedtools

def get_data(ts, ind, t_archaic, windowsize, func, mask=None, chrom=None):
    genome_length = ts.sequence_length
    m = int(genome_length / windowsize) + int(genome_length % windowsize > 0)
    ncoal_sub = np.zeros((len(ind), m))
    t1s_sub = np.zeros((len(ind), m))
    t2s_sub = np.zeros((len(ind), m))
    nleaves_sub = np.zeros((len(ind), m))
    hmm = TRACE()
    tncoal, tt1s, tt2s, treespan, tnleaves = hmm.prepare_data_tmrca(ts = ts, ind = ind, t_archaic=t_archaic)
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
        while t < treespan.shape[0] and treespan[t][0] < int(k * windowsize + windowsize):
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
                    treelens.append(min(treespan[curtrees[j]][1], int(k * windowsize + windowsize)) - max(treespan[curtrees[j]][0], int(k * windowsize)))
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
                        ncoal_sub[i][k] = np.average(tncoal[i][curtrees], weights=treelens)
                        t1s_sub[i][k] = np.average(tt1s[i][curtrees], weights=treelens)
                        t2s_sub[i][k] = np.average(tt2s[i][curtrees], weights=treelens)
                        nleaves_sub[i][k] = np.average(tnleaves[i][curtrees], weights=treelens)
            curtrees = []
            if treespan[t-1][1] < (k+1) * windowsize + windowsize:
                if mask[t-1] == 1:
                    curtrees.append(t - 1)
                else:
                    curtrees.append(-1)
    return ncoal_sub, t1s_sub, t2s_sub, nleaves_sub, treespan, accessible_windows, mask

def main():
    parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--tree-file",
        help="input one tree file, end with .trees or .tsz",
        type=str,
        default=None,
    )
    required.add_argument(
        "--t-archaic",
        help='the time defining the "archaic" population (e.g. the split time between the archaic and modern human'
        + ' populations should be older than this time), in generations',
        type=int,
        required=True,
    )
    required.add_argument(
        "--outpref",
        help="prefix for output file, output file will be named as [outpref].npz",
        type=str,
        required=True,
    )
    mutual_exclusive = parser.add_argument_group("one of which is required")
    me = mutual_exclusive.add_mutually_exclusive_group(required=True)
    me.add_argument(
        "--individual-file",
        help="a list of individual names or id to run the HMM on, take tree node ID as default, only take"
        + "sample name if --sample-names is specified, one individual per line, no header",
        type=str,
        default=None,
    )
    me.add_argument(
        "--individuals",
        help="a list of individual names or id to run the HMM on, take tree node ID as default, only take" 
        + "sample name if --sample-names is specified, specify as --individuals ind1,ind2,ind3",
        type=str,
        default=None,
    )
    optional = parser.add_argument_group("optional")
    optional.add_argument(
        "--window-size",
        help="window size summarizing singer tree sequence",
        type=int,
        default=1000,
    )
    optional.add_argument(
        "--sample-names",
        help="a file containing sample names for all individuals in the tree sequence, " +
        "tab separated, two columns, first column contains tree node id (int), " + 
        "second column contains sample names (str)",
        type=str,
        default=None,
    )
    optional.add_argument(
        "--func",
        help="function to summarize the data in certain windows, can be mean or median",
        type=str,
        default="mean",
    )
    optional.add_argument(
        "--chrom",
        help="chromosome ID for the tree sequence, must match the chromosome ID in the include regions file",
        type=str,
        default=None,
    )
    optional.add_argument(
        "--include-regions",
        help="a BED file containing the INCLUDE regions for the tree sequence",
        type=str,
        default=None,
    )
    optional.add_argument(
        "--mutation-age",
        help="only extract mutation ages in the tree sequence, limited by include regions if specified",
        action="store_true",
        default=False,
    )
    
    args = parser.parse_args()

    # read the args
    tree_file = args.tree_file
    indiv = args.individuals
    indiv_file = args.individual_file
    sample_names = args.sample_names
    t_archaic = args.t_archaic
    outpref = args.outpref
    s = args.window_size
    func = args.func
    chrom = args.chrom
    include_regions = args.include_regions
    mutation_age = args.mutation_age

    # check the function
    if func == "mean":
        func = np.mean
    elif func == "median":
        func = np.median
    else:
        print(f"Unrecognized function: {func}")
        sys.exit(1)

    # read in tree file
    try:
        print(f"loading {tree_file}")
        if tree_file.endswith(".trees"):
            ts = tskit.load(tree_file)
        elif tree_file.endswith(".tsz"):
            ts = tszip.decompress(tree_file)
        else:
            print(f"Unrecognized file extension: {tree_file}")
            sys.exit(1)
    except FileNotFoundError:
        print(f"could not find {tree_file}.")
        sys.exit(1)

    if mutation_age:
        out = ""
        for tree in ts.trees():
            for mut in tree.mutations():
                if tree.parent(mut.node) != tskit.NULL:
                    out += f"{chrom}\t{int(ts.site(mut.site).position) - 1}\t{int(ts.site(mut.site).position)}\t{tree.time(mut.node)}_{tree.time(tree.parent(mut.node))}\n"
                else:
                    out += f"{chrom}\t{int(ts.site(mut.site).position) - 1}\t{int(ts.site(mut.site).position)}\t{tree.time(mut.node)}_{tree.time(mut.node)}\n"
        a = pybedtools.BedTool(out, from_string=True)
        if include_regions is not None:
            print(f"loading {include_regions}")
            include_regions = pybedtools.BedTool(include_regions)
            a = a.intersect(include_regions, u=True)
        out = "chromosome\tposition\tmutation_age\n"
        for x in a:
            out += f"{x.chrom}\t{x.end}\t{x[3]}\n"
        with open(f"{outpref}.txt", 'w') as f:
            f.write(out)
        print(f"mutation ages saved to {outpref}.txt")
        sys.exit(0)
    
    # handle sample names
    if indiv is not None:
        indiv = indiv.strip("\"'").strip(",").split(',')
    else:
        try:
            with open(indiv_file, 'r') as f:
                indiv = f.readlines()
            indiv = [x.strip() for x in indiv]
        except FileNotFoundError:
            print(f"could not find {indiv_file}.")
            sys.exit(1)
    try:
        indiv = [int(x) for x in indiv if len(x) > 0]
    except:
        indiv = [str(x) for x in indiv if len(x) > 0]
    output_utils = Output_utils(samplefile = sample_names, samplename = indiv)
    if sample_names is not None:
        samplename_to_tsid, tsid_to_samplename = output_utils.read_samplename()
        if isinstance(indiv[0], str):
            indiv = np.array([samplename_to_tsid[x] for x in indiv])
    # makesure indiv is tree node ID
    assert isinstance(indiv[0], int)

    # handle include regions
    if include_regions is not None:
        print(f"loading {include_regions}")
        try:
            print(f"chromosome ID: {chrom}")
        except:
            print(f"chromosome ID is not specified.")
            sys.exit(1)
    
    # get data
    m = int(ts.sequence_length / s) + int(ts.sequence_length % s > 0)
    ncoal = np.zeros((len(indiv), m))
    t1s = np.zeros((len(indiv), m))
    t2s = np.zeros((len(indiv), m))
    ncoal, t1s, t2s, nleaves, treespan, accessible_windows, mask = get_data(ts, indiv, t_archaic, s, func, include_regions, chrom)
    atreespan = np.array([[t*s, (t + 1) * s] for t in range(m)])
    # save as npz file
    np.savez_compressed(
        f"{outpref}.npz", 
        ncoal = ncoal,
        t1s = t1s,
        t2s = t2s,
        nleaves = nleaves,
        marginal_treespan = treespan,
        treespan = atreespan,
        marginal_mask = mask,
        accessible_windows = accessible_windows,
        individuals = indiv,
    )

main()
