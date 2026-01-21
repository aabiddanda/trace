"""Utility functions for introgression in Tree-Sequences."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tskit
from intervaltree import Interval, IntervalTree
from numpy import mean, median, var
from scipy.optimize import differential_evolution, minimize
from scipy.stats import gamma
from tqdm import tqdm


class PerformanceUtils:
    """Utilities for performance inference from intervals."""

    def __init__(
        self,
        pp=None,
        treespan=None,
        nodes=None,
    ):
        """Initialize the performance utils class."""
        self.treespan = treespan
        self.pp = pp
        self.nodes = nodes

    def calculate_performance(self, i, t):
        """Calculate recall and precision from tuple objects that are passed in.

        The inferred and truth lists are lists of tuples.
        """
        inferred = IntervalTree.from_tuples(i)
        inferred.merge_overlaps()
        truth = IntervalTree.from_tuples(t)
        truth.merge_overlaps()
        total_HMM = sum([x[1] - x[0] for x in (inferred)])
        truth_seq = sum([x[1] - x[0] for x in (truth)])
        true_positives = []
        false_discovery = []
        for x in inferred:
            overlap = truth.overlap(x)
            if len(overlap) > 0:
                for seg in list(overlap):
                    true_positives.append(min([seg[1], x[1]]) - max([seg[0], x[0]]))
            else:
                false_discovery.append(x[1] - x[0])
        if total_HMM == 0:
            precision = np.nan
        else:
            precision = sum(true_positives) / float(total_HMM)
        if truth_seq == 0:
            recall = np.nan
        else:
            recall = sum(true_positives) / float(truth_seq)
        return (
            precision,
            recall,
            true_positives,
            false_discovery,
            truth_seq,
            total_HMM,
        )

    def read_truth_bed(self, file, popsize):
        """Read truth from bed file."""
        infile = open(file)
        lines = infile.readlines()
        infile.close()
        truth = dict({i: [] for i in range(popsize)})
        popmerge = False
        if len(lines) > 0:
            for i in range(len(lines)):
                s = lines[i].strip("\n").strip("\t").split("\t")
                if len(s) > 3:
                    if not float(s[1]) == float(s[2]) and int(s[3]) in range(popsize):
                        truth[int(s[3])].append([float(s[1]), float(s[2])])
                else:
                    popmerge = True
                    if not float(s[1]) == float(s[2]) and int(s[3]) in range(popsize):
                        truth[0].append([float(s[1]), float(s[2])])
        for i in range(popsize):
            truth[i] = np.array(truth[i])
        if popmerge:
            truth = truth[0]
        return truth

    def read_ind_nodes(self, nodes, idx, popsize, total_tree=None):
        """Read numpy array of node id for potential introgressing nodes (the first node < t_admix for the focal lineage) of one individual.

        nodes: 1-d array of node ids for one individual.

        return: numpy array for node id, each row should be one individual, each column should be node id for one marginal tree.
        """
        if self.nodes is None:
            if total_tree is None:
                self.nodes = np.zeros(shape=(popsize, len(self.treespan)))
            else:
                self.nodes = np.zeros(shape=(popsize, total_tree))
        self.nodes[idx] = nodes
        return self.nodes

    def maxlen(self, states, pp_cutoff, method="posterior", pp=None, nodes=None):
        """Apply maximum length to recover short segments based on introgression node / posterior prob information.

        method:
        'posterior' -- decide the treespan / block to be introgressed for the focal individual if any individual detects introgression at the region and the local pp > pp_cutoff.
        'nodes' -- decide the treespan / block to be introgressed for the focal individual if any individual detects introgression at the region and it has the same introgressing node.
        states: numpy array of 0/1 indicating states for different individuals at different treespans.

        return numpy array of 0/1.
        """
        if pp is None:
            pp = self.pp
        if nodes is None:
            nodes = self.nodes
        maxlen_out = states.copy()
        if method == "nodes":
            for k in range(states.shape[0]):
                for j in range(states.shape[1]):
                    if states[k][j] == 1:
                        n = nodes[k][j]
                        for i in range(nodes.shape[0]):
                            if nodes[i][j] == n:
                                maxlen_out[i][j] = 1
        if method == "posterior":
            for k in range(states.shape[0]):
                for j in range(states.shape[1]):
                    if states[k][j] == 1:
                        for i in range(pp.shape[0]):
                            if pp[i][j] >= pp_cutoff:
                                maxlen_out[i][j] = 1
        self.maxlen_out = maxlen_out
        return maxlen_out

    def get_filtered_tracts(self, states, treespan):
        """Read states matrix and get filtered tracts based on treespan."""
        out = []
        for i in range(states.shape[0]):
            s = states[i]
            j = 0
            ind_out = []
            while j < len(s):
                if s[j] == 1:
                    t = j
                    temp_pos = []
                    while t < len(s) and s[t] == 1:
                        temp_pos.append(t)
                        t += 1
                    ind_out.append(
                        (
                            treespan[np.min(temp_pos)][0],
                            treespan[np.max(temp_pos)][1],
                        )
                    )
                    j = t
                else:
                    j += 1
            out.append(ind_out)
        return out

    def filter_hmm_output(
        self,
        arc_cutoff=0.5,
        pp_cutoff=0.9,
        l_cutoff=0.03,
        popmerge=False,
        maxlen=None,
        combined_pp=None,
        treespan=None,
    ):
        """Filter HMM output based on pp cutoff and length cutoff for each chromosome, all individuals.

        combined_pp: numpy array for posterior probabilities, each row should be an
        individual, each column should be posterior probability for a tree.

        treespan: a numpy array, each row should be a tree, the first column should be
        tree.left.interval, the second column should be tree.right.interval. Trees should be
        ordered in increasing order (or default order returned by ts.trees()).

        maxlen:
        None -- do not apply maxlen filters
        'posterior' -- see maxlen
        'nodes' -- see maxlen

        popmerge: if True, would return a list of tuples merging all archaic regions in the
        population; if False, would return a list of lists of tuples, each list represent
        archaic regions in an individual.
        """
        out = []
        states = np.zeros(shape=(combined_pp.shape[0], combined_pp.shape[1]))
        # apply pp_cutoff and l_cutoff to get states
        for k in range(combined_pp.shape[0]):
            pp = combined_pp[k]
            i = 0
            while i < combined_pp.shape[1]:
                if pp[i] >= arc_cutoff:
                    j = i
                    temp_pos = []
                    temp_pp = []
                    while j < len(pp) and pp[j] >= arc_cutoff:
                        temp_pos.append(j)
                        temp_pp.append(pp[j])
                        j += 1
                    if (
                        np.mean(temp_pp) >= pp_cutoff
                        and treespan[np.max(temp_pos)][1]
                        - treespan[np.min(temp_pos)][0]
                        >= l_cutoff
                    ):
                        for t in temp_pos:
                            states[k][t] = 1
                    i = j
                else:
                    i += 1
        # apply maxlen filters
        if maxlen is None:
            out = self.get_filtered_tracts(states, treespan)
        else:
            maxlen_out = self.maxlen(
                method=maxlen,
                states=states,
                pp_cutoff=pp_cutoff,
                pp=combined_pp,
                nodes=None,
            )
            out = self.get_filtered_tracts(maxlen_out, treespan)
        # return output
        if popmerge:
            output = out[0]
            for i in range(1, len(out)):
                output = output + out[i]
            tree = IntervalTree.from_tuples(output)
            tree.merge_overlaps()
            tree = list(sorted(tree))
            output = []
            for i in range(len(tree)):
                output.append((tree[i][0], tree[i][1]))
            return output
        else:
            return out, states


class OutputUtils:
    """Utility to filter and generate hmm output files."""

    def __init__(
        self,
        samplefile=None,
        samplename=None,
    ):
        """Initialize the ARG-output class."""
        self.samplefile = samplefile
        self.samplename = samplename

    def read_samplename(self):
        """Read from samplename file, return two dictionaries."""
        assert self.samplefile is not None
        samplename_to_tsid = {}
        tsid_to_samplename = {}
        print(f"Reading sample name file {self.samplefile}")
        infile = open(self.samplefile)
        lines = infile.readlines()
        infile.close()
        for i in range(len(lines)):
            s = lines[i].strip("\n").split()  # split by tab or space, check this
            if len(s) >= 2:
                samplename_to_tsid[str(s[1])] = int(s[0])
                tsid_to_samplename[int(s[0])] = str(s[1])
            elif len(s) == 0 and i == len(lines) - 1:
                continue
            else:
                print("Error: empty row")
                sys.exit(0)
        return samplename_to_tsid, tsid_to_samplename

    def filter_tracts(
        self,
        indiv_pp,
        treespan,
        pp_cutoff=0.9,
    ):
        tracts = []
        # apply pp_cutoff to get states
        i = 0
        while i < indiv_pp.shape[1]:
            if indiv_pp[1][i] >= pp_cutoff:
                j = i
                temp_pos = []
                temp_pp = []
                while j < len(indiv_pp[1]) and indiv_pp[1][j] >= pp_cutoff:
                    temp_pos.append(j)
                    temp_pp.append(indiv_pp[1][j])
                    j += 1
                tracts.append(
                    [
                        treespan[temp_pos[0]][0],
                        treespan[temp_pos[-1]][1],
                        np.mean(temp_pp),
                        "Archaic",
                    ]
                )
            else:
                j = i
                temp_pos = []
                temp_pp = []
                while j < len(indiv_pp[1]) and indiv_pp[1][j] < pp_cutoff:
                    temp_pos.append(j)
                    temp_pp.append(indiv_pp[0][j])
                    j += 1
                tracts.append(
                    [
                        treespan[temp_pos[0]][0],
                        treespan[temp_pos[-1]][1],
                        np.mean(temp_pp),
                        "Human",
                    ]
                )
            i = j
        return tracts

    def summarize(
        self,
        files,
        cleanup=True,
        pp_cutoff=0.9,
    ):
        """Summarize all pp results, do posterior cutoff based on thresholds shown."""
        assert self.samplename is not None
        # Check if one has combined results in files
        # if cleanup, remove original separate files

        # read pp matrix, 0 is human, 1 is archaic, for each chromosome separately
        out = "chromosome\tstart\tend\tmean_posterior\tstate\n"
        pp_human = []
        pp_archaic = []
        treespan = []
        i = 1
        s = lines[i].strip("\n").split("")  # split by tab or space, check this
        cur_chrom = str(s[0])
        for i in range(1, len(lines)):
            s = lines[i].strip("\n").split("")  # split by tab or space, check this
            if str(s[0]) != cur_chrom:
                pp = np.array([pp_human, pp_archaic])
                treespan = np.array(treespan)
                tracts = self.filter_tracts(pp, treespan, pp_cutoff)
                for seg in tracts:
                    out += cur_chrom + "\t" + str(seg[0]) + "\t" + str(seg[1]) + "\t"
                    out += str(seg[2]) + "\t" + str(seg[3]) + "\n"
                cur_chrom = str(s[0])
                treespan = []
                pp_archaic = []
                pp_human = []
            treespan.append([float(s[1]), float(s[2])])
            pp_archaic.append(float(s[3]))
            pp_human.append(float(s[4]))
        outfile = open(str(self.samplename) + ".summary.txt", "w")
        outfile.write(out)
        outfile.close()


class ExplicitDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    format the help menu such that defaults are printed to help
    only when they are explicitly stated in the parser
    (e.g. do not print actions or Nones)
    """

    def _get_help_string(self, action):
        """
        returns the help string
        """

        if action.default in (None, False):
            return action.help
        return super()._get_help_string(action)

