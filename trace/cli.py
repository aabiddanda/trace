"""CLI for ghosthmm."""
import logging
import sys

import click
import numpy as np
import pandas as pd

from ghosthmm import GhostProductHMM

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input data file for PGT-A array intensity data.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="karyohmm",
    help="Output file prefix.",
)
def main(
    input,
    out="ghosthmm",
):
    """GhostHMM-Inference CLI."""
    logging.info(f"Starting to read input data {input}...")
