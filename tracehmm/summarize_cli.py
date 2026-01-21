"""CLI for TRACE."""
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


@click.command()
@click.option(
    "--mode",
    "-m",
    required=True,
    multiple=False,
    type=click.Choice(["extract", "infer", "summarize"]),
    help="Mode.",
)
@click.option(
    "--input",
    "-i",
    required=False,
    type=click.Path(exists=True),
    help="Input data in tskit format.",
)
@click.option(
    "--time",
    "-t",
    required=False,
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
    "--out",
    "-o",
    required=True,
    type=str,
    default="trace",
    help="Output file prefix.",
)
def main(
    mode="extract",
    input=None,
    time=15e3,
    out="trace",
):
    """TRACE-Inference CLI."""
    logging.info(f"Starting TRACE in mode {mode}...")
    if mode == "extract":
        pass
    if mode == "infer":
        pass
    if mode == "summarize":
        pass
