import numpy as np
import tskit
import tszip
import sys
from trace.trace import TRACE
from trace.utils import (
    ExplicitDefaultsHelpFormatter,
    Output_utils
)
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(formatter_class=ExplicitDefaultsHelpFormatter)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--individual",
        help="the focal individual name or id to run the HMM on, take tree node ID as default, only take" 
        + "sample name if --sample-names is specified",
        required=True,
    )
    mutual_exclusive = parser.add_argument_group("one of which is required")
    me = mutual_exclusive.add_mutually_exclusive_group(required=True)
    me.add_argument(
        "--npz-file",
        help="input one npz file",
        type=str,
        default=None,
    )
    me.add_argument(
        "--data-file",
        help="a list of .npz/npy files, output from getdata_singer.py",
        type=str,
        default=None,
    )
    optional = parser.add_argument_group("optional")
    optional.add_argument(
        "--chrom",
        help="chromosome ID used in the output file, int",
        type=str,
        default="1",
    )
    optional.add_argument(
        "--subrange",
        help="a subrange of treesequence to run GhostHMM on, specify as --subrange lowerEdge,upperEdge, use " 
        + "the whole tree sequence as default",
        type=str,
        default=None,
    )
    optional.add_argument(
        "--seed",
        help="set the random seed",
        type=int,
        default=42,
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
        "--genetic-map",
        help="a HapMap format genetic map (see https://ftp.ncbi.nlm.nih.gov/hapmap/recombination/2011-01_phaseII_B37/ for hg19 HapMap genetic map)," + 
        "the 2nd and 4th column (1-index) should be position (bp) and genetic distance (cM); assume a uniform recombination rate of 1e-8 per" + 
        " bp per generation if not specified",
        type=str,
        default=None,
    )
    optional.add_argument(
        "--outpref",
        help="prefix for output file, output file will be named as [outpref][individual].[chrom]_[subrange].xss.npz",
        type=str,
        default=None,
    )
    # need to fix this (by default check the outlier code setting)
    optional.add_argument(
        "--proportion-admix",
        help="prior probability of admixture, default is 0.01",
        type=float,
        default=None,
    )
    optional.add_argument(
        "--func",
        help="function to summarize across SINGER runs, could be mean or median",
        type=str,
        default="mean",
    )
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)

    # read the args
    chroms = str(args.chrom).strip(",").split(",")
    subrange = args.subrange
    npz_file = args.npz_file
    data_file = args.data_file
    indiv = args.individual
    sample_names = args.sample_names
    t_archaic = args.t_archaic
    genetic_map = args.genetic_map
    outpref = str(args.outpref).strip(",").split(",")
    intro_prop = args.proportion_admix
    include_regions = args.include_regions
    func = args.func
    if func not in ["mean", "median"]:
        print(f"Unrecognized function: {func}, use mean as default")
    elif func == "mean":
        func = np.ma.mean
    else:
        func = np.ma.median
    
    # handle sample names
    try:
        indiv = int(indiv)
    except:
        indiv = str(indiv)
    output_utils = Output_utils(samplefile = sample_names, samplename = indiv)
    if sample_names is not None:
        samplename_to_tsid, tsid_to_samplename = output_utils.read_samplename()
        if isinstance(indiv, str):
            indiv = samplename_to_tsid[indiv]
    # makesure indiv is tree node ID
    assert isinstance(indiv, int)

    # handle subrange
    if subrange is not None:
        subrange = subrange.strip("\"'").strip(',').split(',')
        subrange = [int(x) for x in subrange]

    hmm = TRACE()
    # load data
    if npz_file is not None:
        datafiles = str(npz_file).strip(",").split(",")
        assert len(chroms) == len(datafiles), "number of chromosomes and data files must be the same"
        chromfile_edges = []
        for idx, data_file in enumerate(datafiles):
            print(f"loading {data_file}")
            data = np.load(data_file)
            individuals = data['individuals']
            indiv_idx = np.where(individuals == indiv)[0][0]
            oncoal = data['ncoal'][indiv_idx]
            ot1s = data['t1s'][indiv_idx]
            ot2s = data['t2s'][indiv_idx]
            onleaves = data['nleaves'][indiv_idx]
            oinclude_regions = data['accessible_windows']
            masked_ncoal = np.ma.masked_array(oncoal, mask=(oinclude_regions == 0))
            masked_t1s = np.ma.masked_array(ot1s, mask=(oinclude_regions == 0))
            masked_t2s = np.ma.masked_array(ot2s, mask=(oinclude_regions == 0))
            masked_nleaves = np.ma.masked_array(onleaves, mask=(oinclude_regions == 0))
            chromfile_edges.append(data['treespan'].shape[0])
            if idx == 0:
                treespan = data['treespan']
                ncoal = func(masked_ncoal, axis=0).data
                t1s = func(masked_t1s, axis=0).data
                t2s = func(masked_t2s, axis=0).data
                nleaves = func(masked_nleaves, axis=0).data
                include_regions = oinclude_regions
            else:
                treespan = np.vstack((treespan, data['treespan']))
                ncoal = np.concatenate((ncoal, func(masked_ncoal, axis=0).data))
                t1s = np.concatenate((t1s, func(masked_t1s, axis=0).data))
                t2s = np.concatenate((t2s, func(masked_t2s, axis=0).data))
                nleaves = np.concatenate((nleaves, func(masked_nleaves, axis=0).data))
                include_regions = np.concatenate((include_regions, oinclude_regions))
    else:
        datafiles = str(data_file).strip(",").split(",")
        assert len(chroms) == len(datafiles), "number of chromosomes and data files must be the same"
        chromfile_edges = []
        for idx, data_file in enumerate(datafiles):
            with open(data_file, 'r') as f:
                data_files = f.readlines()
            print(f"loading {data_files[0]}")
            data = np.load(data_files[0].strip())
            individuals = data['individuals']
            indiv_idx = np.where(individuals == indiv)[0][0]
            oncoal = data['ncoal'][indiv_idx]
            ot1s = data['t1s'][indiv_idx]
            ot2s = data['t2s'][indiv_idx]
            onleaves = data['nleaves'][indiv_idx]
            oinclude_regions = data['accessible_windows']
            for i in range(1, len(data_files)):
                x = data_files[i].strip()
                print(f"loading {x}")
                data = np.load(x)
                individuals = data['individuals']
                indiv_idx = np.where(individuals == indiv)[0][0]
                oncoal = np.vstack((oncoal, data['ncoal'][indiv_idx]))
                ot1s = np.vstack((ot1s, data['t1s'][indiv_idx]))
                ot2s = np.vstack((ot2s, data['t2s'][indiv_idx]))
                onleaves = np.vstack((onleaves, data['nleaves'][indiv_idx]))
                oinclude_regions = np.vstack((oinclude_regions, data['accessible_windows']))
            masked_ncoal = np.ma.masked_array(oncoal, mask=(oinclude_regions == 0))
            masked_t1s = np.ma.masked_array(ot1s, mask=(oinclude_regions == 0))
            masked_t2s = np.ma.masked_array(ot2s, mask=(oinclude_regions == 0))
            masked_nleaves = np.ma.masked_array(onleaves, mask=(oinclude_regions == 0))
            chromfile_edges.append(data['treespan'].shape[0])
            if idx == 0:
                treespan = data['treespan']
                ncoal = func(masked_ncoal, axis=0).data
                t1s = func(masked_t1s, axis=0).data
                t2s = func(masked_t2s, axis=0).data
                nleaves = func(masked_nleaves, axis=0).data
                include_regions = np.max(oinclude_regions, axis=0)
            else:
                treespan = np.vstack((treespan, data['treespan']))
                ncoal = np.concatenate((ncoal, func(masked_ncoal, axis=0).data))
                t1s = np.concatenate((t1s, func(masked_t1s, axis=0).data))
                t2s = np.concatenate((t2s, func(masked_t2s, axis=0).data))
                nleaves = np.concatenate((nleaves, func(masked_nleaves, axis=0).data))
                include_regions = np.concatenate((include_regions, np.max(oinclude_regions, axis=0)))
    
    # run hmm
    hmm.init_hmm(ncoal, treespan, intro_prop=intro_prop, subrange=subrange, include_regions=include_regions)
    if genetic_map is not None:
        gmaps = str(genetic_map).strip(",").split(",")
        assert len(chroms) == len(gmaps), "number of chromosomes and genetic map files must be the same"
        for idx, gmap in enumerate(gmaps):
            start = 0 if idx == 0 else np.sum(chromfile_edges[:idx])
            end = np.sum(chromfile_edges[:(idx + 1)])
            hmm.treespan[start:end] = hmm.add_recombination_map(treespan[start:end], gmap)
    print(f"mean e_null: {hmm.emi2_a1 / hmm.emi2_b1}, std e_null: {np.sqrt(hmm.emi2_a1 / (hmm.emi2_b1 ** 2))}")
    print(f"mean e_alt: {hmm.emi2_a2 / hmm.emi2_b2}, std e_alt: {np.sqrt(hmm.emi2_a2 / (hmm.emi2_b2 ** 2))}")
    res_dict = hmm.train(seed=args.seed)
    outparams = pd.DataFrame.from_dict(res_dict).to_numpy()
    print(f"emi2_a1: {hmm.emi2_a1}, emi2_b1: {hmm.emi2_b1}, emi2_a2: {hmm.emi2_a2}, emi2_b2: {hmm.emi2_b2}")
    print(f"mean e2: {hmm.emi2_a1 / hmm.emi2_b1}, std e2: {np.sqrt(hmm.emi2_a1 / (hmm.emi2_b1 ** 2))}, mean e2: {hmm.emi2_a2 / hmm.emi2_b2}, std e2: {np.sqrt(hmm.emi2_a2 / (hmm.emi2_b2 ** 2))}")
    gammas, _, _ = hmm.decode()

    # save hmm.xss as npz file
    if sample_names is not None:
        indiv = tsid_to_samplename[indiv]
    for idx, chrom in enumerate(chroms):
        if len(outpref) == 1:
            outp = outpref[0]
        else:
            outp = outpref[idx]
        if subrange is None:
            # outname = f"{outp}{indiv}.{chrom}.xss.npz"
            outname = f"{outp}.{chrom}.xss.npz"
        else:
            outname = f"{outp}{indiv}.{chrom}_{subrange[0]}_{subrange[1]}.xss.npz"
        start = 0 if idx == 0 else np.sum(chromfile_edges[:idx])
        end = np.sum(chromfile_edges[:(idx + 1)])
        print(f"saving to {outname}")
        np.savez_compressed(
            outname,
            t1s = t1s[start:end],
            t2s = t2s[start:end],
            nleaves = nleaves[start:end],
            ncoal = ncoal[start:end],
            treespan = hmm.treespan[start:end],
            treespan_phy = hmm.treespan_phy[start:end],
            func = args.func,
            accessible_windows = include_regions[start:end],
            params = outparams,
            gammas = np.exp(gammas[:, start:end])
        )
main()