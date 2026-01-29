"""Utility functions for introgression in Tree-Sequences."""

import sys

import numpy as np


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
            s = lines[i].strip("\n").split()
            if i == 0:
                try:
                    treenode_id = int(s[0])
                except (ValueError, IndexError, KeyError):
                    print("Skipping first row (header or invalid format)")
                    continue  # Skip to next row
            try:
                treenode_id = int(s[0])
                idname = str(s[1])
            except ValueError:
                print(f"{s[0]} is not a valid tree node ID ... exiting.")
                sys.exit(1)
            if len(s) >= 2:
                samplename_to_tsid[idname] = treenode_id
                tsid_to_samplename[treenode_id] = idname
            elif len(s) == 0 and i == len(lines) - 1:
                continue
            else:
                print(f"Error: empty row or invalid format at row {i} ... exiting.")
                sys.exit(1)
        return samplename_to_tsid, tsid_to_samplename

    def filter_tracts(
        self,
        indiv_pp,
        treespan,
        treespan_phy,
        pp_cutoff=0.9,
        arc_cutoff=0.5,
        phy_cutoff=5e4,
        l_cutoff=0.05,
    ):
        """Filter estimated tracts from TRACE based on external criteria."""
        tracts = []
        states = np.zeros(indiv_pp.shape[1])
        i = 0
        while i < indiv_pp.shape[1]:
            if indiv_pp[1][i] >= pp_cutoff:
                j = i
                temp_pos = []
                temp_pp = []
                while j < len(indiv_pp[1]) and indiv_pp[1][j] >= arc_cutoff:
                    temp_pos.append(j)
                    temp_pp.append(indiv_pp[1][j])
                    j += 1
                if (
                    np.mean(temp_pp) >= pp_cutoff
                    and treespan[np.max(temp_pos)][1] - treespan[np.min(temp_pos)][0]
                    >= l_cutoff
                    and treespan_phy[np.max(temp_pos)][1]
                    - treespan_phy[np.min(temp_pos)][0]
                    >= phy_cutoff
                ):
                    start = treespan_phy[temp_pos[0]][0]
                    end = treespan_phy[temp_pos[-1]][1]
                    tracts.append(
                        [
                            start,
                            end,
                            np.mean(temp_pp),
                            end - start,
                            treespan[temp_pos[-1]][1] - treespan[temp_pos[0]][0],
                        ]
                    )
                    states[np.array(temp_pos)] = 1
                i = j
            else:
                i += 1
        return tracts, states

    def summarize(
        self,
        pp,
        treespan,
        treespan_phy,
        outpref,
        chrom="chr1",
        pp_cutoff=0.9,
        phy_cutoff=5e4,
        l_cutoff=0.05,
    ):
        """Summarize all pp results, do posterior cutoff based on thresholds shown."""
        out = ""
        tracts, states = self.filter_tracts(
            pp,
            treespan,
            treespan_phy,
            pp_cutoff=pp_cutoff,
            phy_cutoff=phy_cutoff,
            l_cutoff=l_cutoff,
        )
        for i in range(len(tracts)):
            out += (
                chrom
                + "\t"
                + str(int(tracts[i][0]))
                + "\t"
                + str(int(tracts[i][1]))
                + "\t"
                + str(round(tracts[i][2], 2))
                + "\t"
                + str(int(tracts[i][3]))
                + "\t"
                + str(round(tracts[i][4], 3))
                + "\n"
            )
        return out, states
