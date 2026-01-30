"""CLI for TRACE-Infer."""
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from tracehmm import TRACE

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command(context_settings={"show_default": True})
@click.option(
    "--individual",
    "-i",
    required=True,
    type=str,
    help="the focal individual tree node id to run the HMM on, only take a "
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
    help="a list of .npz files (outputs from trace-extract), one file per line."
    + " If multiple chromosomes are provided, provide one data file per chromosome, separated by comma (no spaces).",
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
    required=False,
    type=click.Choice(["mean", "median"]),
    default="mean",
    help="Summarize function for windows across posterior tree sequences.",
)
@click.option(
    "--genetic-maps",
    help="a HapMap formatted genetic map (see https://ftp.ncbi.nlm.nih.gov/hapmap/recombination/2011-01_phaseII_B37/ for hg19 HapMap genetic map),"
    + "the 2nd and 4th column (1-index) should be position (bp) and genetic distance (cM); if multiple chromosomes are provided, "
    + "separate by comma (no spaces). "
    + "assume a uniform recombination rate of 1e-8 per bp per generation if not specified",
    type=str,
    default=None,
)
@click.option(
    "--seed",
    help="random seed ",
    type=int,
    default=42,
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
    genetic_maps,
    seed,
    out="trace",
):
    """TRACE-Inference CLI."""
    logging.info("Starting trace-infer...")
    logging.info(f"Setting random seed to {seed} ...")
    chroms = str(chroms).strip(",").split(",")
    logging.info(f"Analysis to be run for {chroms} ...")
    if func not in ["mean", "median"]:
        logging.info(f"Unrecognized function: {func} for aggregation")
    elif func == "mean":
        func = np.ma.mean
    else:
        func = np.ma.median

    logging.info("Establishing sample IDs ...")
    # handle sample names
    try:
        indiv = int(individual)
    except ValueError:
        logging.info(f"Cannot convert individual {individual} to int ... exiting.")
        sys.exit(1)
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
            if Path(data_file).is_file():
                data = np.load(data_file)
            else:
                logging.info(f"Listed {data_file} is not a file ... exiting.")
                sys.exit(1)
            individuals = data["individuals"]
            if indiv not in individuals:
                logging.info(
                    f"Individual {indiv} not found in data file {data_file} ... exiting."
                )
                sys.exit(1)
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
            if Path(data_file).is_file():
                logging.info(f"loading {data_file} ...")
                with open(data_file, "r") as f:
                    data_files = f.readlines()
            else:
                logging.info(
                    f"File {data_file} is not a valid filepath from `--data-files`  ... exiting."
                )
                sys.exit(1)
            for fp in data_files:
                if not Path(fp.strip()).is_file():
                    logging.info(
                        f"File {fp} from {data_file} is not a valid filepath ... exiting."
                    )
                    sys.exit(1)
            data = np.load(data_files[0].strip())
            individuals = data["individuals"]
            if indiv not in individuals:
                logging.info(
                    f"Individual {indiv} not found in data file {data_files[0]} ... exiting."
                )
                sys.exit(1)
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
                if indiv not in individuals:
                    logging.info(
                        f"Individual {indiv} not found in data file {x} ... exiting."
                    )
                    sys.exit(1)
                indiv_idx = np.where(individuals == indiv)[0][0]
                try:
                    oncoal = np.vstack((oncoal, data["ncoal"][indiv_idx]))
                    ot1s = np.vstack((ot1s, data["t1s"][indiv_idx]))
                    ot2s = np.vstack((ot2s, data["t2s"][indiv_idx]))
                    oinclude_regions = np.vstack(
                        (oinclude_regions, data["accessible_windows"])
                    )
                except ValueError:
                    logging.info(
                        f"inconsistent data dimensions in {x} and previous files ... exiting."
                    )
                    logging.info(
                        "Please run trace-extract with --window-size specified to ensure consistent"
                        + " data dimensions across posterior tree sequences."
                    )
                    sys.exit(1)
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
    hmm.init_hmm(
        ncoal,
        treespan,
        include_regions=include_regions,
        seed=seed,
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
            logging.info(f"Adding recombination map from {gmap} ...")
            skiprow = True
            with open(gmap, "r") as f:
                first_line = f.readline()
                first_line = first_line.strip().split()
                if first_line[0] == chroms[idx]:
                    skiprow = False
            gmap_df = pd.read_csv(
                gmap,
                sep="\s+",
                header=None,
                skiprows=int(skiprow),
                names=["chrom", "pos", "rate", "gen_dist"],
            )
            assert (
                gmap_df["chrom"].nunique() == 1
            ), "Genetic map file must contain only one chromosome."
            assert gmap_df["chrom"].unique()[0] == chroms[idx], (
                "Chromosome ID in genetic map file must match the one provided in --chroms.\n"
                f"Provided chromosome ID: {chroms[idx]}, chromosome ID in genetic map file: {gmap_df['chrom'].unique()[0]}"
            )
            try:
                gmap_df["pos"] = gmap_df["pos"].astype(float)
                gmap_df["gen_dist"] = gmap_df["gen_dist"].astype(float)
            except (KeyError, ValueError):
                logging.info(
                    f"Position or genetic distance column in genetic map file {gmap} contains non-numeric values ... exiting."
                )
                sys.exit(1)
            assert gmap_df[
                "pos"
            ].is_monotonic_increasing, f"Position column in genetic map file {gmap} must be sorted in increasing order ..."
            start = 0 if idx == 0 else np.sum(chromfile_edges[:idx])
            end = np.sum(chromfile_edges[: (idx + 1)])
            hmm.treespan[start:end] = hmm.add_recombination_map(
                treespan[start:end], gmap, skiprow=skiprow
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
    gammas, _, _ = hmm.decode(seed=seed)

    # if sample_names is not None:
    #     indiv = tsid_to_samplename[indiv]
    for idx, chrom in enumerate(chroms):
        outname = f"{out}.{chrom}.xss.npz"
        start = 0 if idx == 0 else np.sum(chromfile_edges[:idx])
        end = np.sum(chromfile_edges[: (idx + 1)])
        logging.info(f"Writing output to {outname} ...")
        np.savez_compressed(
            outname,
            ncoal=ncoal[start:end],
            treespan=hmm.treespan[start:end],
            treespan_phy=hmm.treespan_phy[start:end],
            accessible_windows=include_regions[start:end],
            params=outparams,
            gammas=np.exp(gammas[:, start:end]),
            seed=np.array([seed]),
            individual=np.array([indiv]),
        )
