"""CLI for TRACE."""
import logging
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
    "--individual",
    required=True,
    type=str,
    help="the focal individual name or id to run the HMM on, take tree node ID as default, only take"
    + "sample name if --sample-names is specified.",
)
@click.option(
    "--npz-files",
    required=False,
    type=str,
    help="Input data in npz format (output from trace-extract). If multiple chromosomes are provided, "
    + "separate by comma (no spaces).",
    default=None,
)
@click.option(
    "--data-files",
    help="a list of .npz/npy files, outputs from trace-extract, one file per line"
    + " (if multiple chromosomes are provided, provide one data file per chromosome, separated by comma, no space).",
    type=str,
    default=None,
)
@click.option(
    "--chroms",
    help="chromosome ID for the tree sequence, must match the chromosome ID in the include regions file. "
    + "If multiple chromosomes are provided, separate by comma (no spaces).",
    type=str,
    default="chr1",
)
@click.option(
    "--func",
    "-f",
    required=False,
    type=click.Choice(["mean", "median"]),
    default="mean",
    help="Summarize function for windows across posterior tree sequences.",
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
    "--genetic-maps",
    help="a HapMap formatted genetic map (see https://ftp.ncbi.nlm.nih.gov/hapmap/recombination/2011-01_phaseII_B37/ for hg19 HapMap genetic map),"
    + "the 2nd and 4th column (1-index) should be position (bp) and genetic distance (cM); if multiple chromosomes are provided, "
    + "separate by comma (no spaces). "
    + "assume a uniform recombination rate of 1e-8 per bp per generation if not specified",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--seed",
    help="random seed ",
    type=int,
    default=42,
)
@click.option(
    "--proportion-admix",
    help="Prior probability of admixture, default is None.",
    type=float,
    default=None,
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="trace",
    help="Output file prefix, output files will be named as [out].[chrom].xss.npz",
)
def main(
    individual,
    npz_files,
    data_files,
    chroms,
    func,
    sample_names,
    genetic_maps,
    seed,
    proportion_admix,
    out="trace",
):
    """TRACE-Inference CLI."""
    logging.info(f"Starting trace-infer...")
    logging.info(f"Setting random seed to {seed} ...")
    chroms = str(chroms).strip(",").split(",")
    logging.info(f"Analysis to be run for {chroms} ...")
    if func not in ["mean", "median"]:
        logging.info(f"Unrecognized function: {func} for aggregation")
    elif func == "mean":
        func = np.ma.mean
    else:
        func = np.ma.median

    logging.info(f"Establishing sample IDs ...")
    # handle sample names
    try:
        indiv = int(individual)
    except:
        indiv = str(individual)
    output_utils = OutputUtils(samplefile=sample_names, samplename=indiv)
    if sample_names is not None:
        samplename_to_tsid, tsid_to_samplename = output_utils.read_samplename()
        if isinstance(indiv, str):
            indiv = samplename_to_tsid[indiv]
    # makesure indiv is tree node ID
    assert isinstance(indiv, int)

    hmm = TRACE()
    if (data_files is None) and (npz_files is None):
        logging.info(
            "Need either --npz-file or --data-file to be specified... exiting."
        )
        sys.exit(1)
    if npz_files is not None:
        logging.info(f"Loading data from {npz_files}...")
        datafiles = str(npz_files).strip(",").split(",")
        if len(chroms) != len(datafiles):
            logging.info(
                f"Mismatch between datafiles and number of chromosomes: {len(datafiles)} != {len(chroms)} ... exiting."
            )
            sys.exit(1)
        assert len(chroms) == len(datafiles)
        chromfile_edges = []
        for idx, data_file in enumerate(datafiles):
            logging.info(f"loading {data_file} ...")
            data = np.load(data_file)
            individuals = data["individuals"]
            indiv_idx = np.where(individuals == indiv)[0][0]
            oncoal = data["ncoal"][indiv_idx]
            ot1s = data["t1s"][indiv_idx]
            ot2s = data["t2s"][indiv_idx]
            oinclude_regions = data["accessible_windows"]
            oncoal[oinclude_regions == 0] = 0
            ot1s[oinclude_regions == 0] = 0
            ot2s[oinclude_regions == 0] = 0
            chromfile_edges.append(data["treespan"].shape[0])
            if idx == 0:
                treespan = data["treespan"]
                ncoal = oncoal
                t1s = ot1s
                t2s = ot2s
                include_regions = oinclude_regions
            else:
                treespan = np.vstack((treespan, data["treespan"]))
                ncoal = np.concatenate((ncoal, oncoal))
                t1s = np.concatenate((t1s, ot1s))
                t2s = np.concatenate((t2s, ot2s))
                include_regions = np.concatenate((include_regions, oinclude_regions))
    else:
        logging.info(f"Running from datafiles {data_files} ...")
        datafiles = str(data_files).strip(",").split(",")
        if len(chroms) != len(datafiles):
            logging.info(
                f"Mismatch between datafiles and number of chromosomes: {len(datafiles)} != {len(chroms)} ... exiting."
            )
            sys.exit(1)
        assert len(chroms) == len(datafiles)
        chromfile_edges = []
        for idx, data_file in enumerate(datafiles):
            logging.info(f"loading {data_file} ...")
            with open(data_file, "r") as f:
                data_files = f.readlines()
            data = np.load(data_files[0].strip())
            individuals = data["individuals"]
            indiv_idx = np.where(individuals == indiv)[0][0]
            oncoal = data["ncoal"][indiv_idx]
            ot1s = data["t1s"][indiv_idx]
            ot2s = data["t2s"][indiv_idx]
            oinclude_regions = data["accessible_windows"]
            for i in range(1, len(data_files)):
                x = data_files[i].strip()
                logging.info(f"loading {x} ...")
                data = np.load(x)
                individuals = data["individuals"]
                indiv_idx = np.where(individuals == indiv)[0][0]
                oncoal = np.vstack((oncoal, data["ncoal"][indiv_idx]))
                ot1s = np.vstack((ot1s, data["t1s"][indiv_idx]))
                ot2s = np.vstack((ot2s, data["t2s"][indiv_idx]))
                oinclude_regions = np.vstack(
                    (oinclude_regions, data["accessible_windows"])
                )
            masked_ncoal = np.ma.masked_array(oncoal, mask=(oinclude_regions == 0))
            masked_t1s = np.ma.masked_array(ot1s, mask=(oinclude_regions == 0))
            masked_t2s = np.ma.masked_array(ot2s, mask=(oinclude_regions == 0))
            chromfile_edges.append(data["treespan"].shape[0])
            if idx == 0:
                treespan = data["treespan"]
                ncoal = func(masked_ncoal, axis=0).data
                t1s = func(masked_t1s, axis=0).data
                t2s = func(masked_t2s, axis=0).data
                include_regions = np.max(oinclude_regions, axis=0)
            else:
                treespan = np.vstack((treespan, data["treespan"]))
                ncoal = np.concatenate((ncoal, func(masked_ncoal, axis=0).data))
                t1s = np.concatenate((t1s, func(masked_t1s, axis=0).data))
                t2s = np.concatenate((t2s, func(masked_t2s, axis=0).data))
                include_regions = np.concatenate(
                    (include_regions, np.max(oinclude_regions, axis=0))
                )

    logging.info("Initializing TRACE ...")
    print(ncoal)
    hmm.init_hmm(
        ncoal,
        treespan,
        intro_prop=proportion_admix,
        include_regions=include_regions,
    )
    if genetic_maps is not None:
        logging.info(f"loading genetic map from {genetic_maps} ...")
        gmaps = str(genetic_maps).strip(",").split(",")
        if len(chroms) != len(gmaps):
            logging.info(
                f"Mismatch between genetic maps and number of chromosomes: {len(gmaps)} != {len(chroms)} ... exiting."
            )
            sys.exit(1)
        assert len(chroms) == len(gmaps)
        for idx, gmap in enumerate(gmaps):
            start = 0 if idx == 0 else np.sum(chromfile_edges[:idx])
            end = np.sum(chromfile_edges[: (idx + 1)])
            hmm.treespan[start:end] = hmm.add_recombination_map(
                treespan[start:end], gmap
            )
    logging.info(
        f"mean e_null: {hmm.emi2_a1 / hmm.emi2_b1}, std e_null: {np.sqrt(hmm.emi2_a1 / (hmm.emi2_b1 ** 2))}"
    )
    logging.info(
        f"mean e_alt: {hmm.emi2_a2 / hmm.emi2_b2}, std e_alt: {np.sqrt(hmm.emi2_a2 / (hmm.emi2_b2 ** 2))}"
    )
    res_dict = hmm.train(seed=seed)
    outparams = pd.DataFrame.from_dict(res_dict).to_numpy()
    logging.info(
        f"emi2_a1: {hmm.emi2_a1}, emi2_b1: {hmm.emi2_b1}, emi2_a2: {hmm.emi2_a2}, emi2_b2: {hmm.emi2_b2}"
    )
    logging.info(
        f"mean e2: {hmm.emi2_a1 / hmm.emi2_b1}, std e2: {np.sqrt(hmm.emi2_a1 / (hmm.emi2_b1 ** 2))}, mean e2: {hmm.emi2_a2 / hmm.emi2_b2}, std e2: {np.sqrt(hmm.emi2_a2 / (hmm.emi2_b2 ** 2))}"
    )
    logging.info("Running TRACE decoding via Forward-Backward algorithm ...")
    gammas, _, _ = hmm.decode()

    if sample_names is not None:
        indiv = tsid_to_samplename[indiv]
    for idx, chrom in enumerate(chroms):
        outname = f"{out}.{chrom}.xss.npz"
        start = 0 if idx == 0 else np.sum(chromfile_edges[:idx])
        end = np.sum(chromfile_edges[: (idx + 1)])
        logging.info(f"Writing output to {outname} ...")
        np.savez_compressed(
            outname,
            # t1s=t1s[start:end],
            # t2s=t2s[start:end],
            ncoal=ncoal[start:end],
            treespan=hmm.treespan[start:end],
            treespan_phy=hmm.treespan_phy[start:end],
            accessible_windows=include_regions[start:end],
            params=outparams,
            gammas=np.exp(gammas[:, start:end]),
        )
