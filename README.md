# TRACE: Estimating Introgression using ancestral recombination graphs

## Installation

In order to install the package locally you can run:

```
git clone git@github.com:aabiddanda/trace.git
cd trace; pip install .
```

## Command-line Interface

TBD

## Running an Example

In order to facilitate a basic example, we have provided some  simulated data to get a feel for using the CLI. 


1. To extract relevant observation data for the HMM, we use:

```
trace-extract --tree-file example_data/n10_seed10_A.tsz -s 0,1,2,3,4,5,6,7,8,9,10 -o example_data/test_output
```

2. To infer a posterior decoding of introgression tracts, we use: 

```
trace-infer --individual 2 --npz-files example_data/test_output.npz -o example_data/test_infer
```

3. To get final archaic introgression tracts, we use:

```
trace-summarize --files example_data/test_infer.chr1.xss.npz --chroms chr1 --out example_data/test_summarize
```


## Interpreting Output Files

1. Output from `trace-extract`: a numpy NpzFile with following numpy arrays
    - Let n be the number of individuals user specified, l be the number of marginal trees (or genomic windows if `--window-size` is specified).
    - "ncoal": (n x l) array storing the observation data for TRACE emission calculation
    - "t1s": (n x l) array storing the lower-end coalescent time for the extracted branch that spans specified `--t-archaic`
    - "t2s": (n x l) array storing the upper-end coalescent time for the extracted branch that spans specified `--t-archaic`
    - "marginal_treespan": (l x 2) array storing the spans of marginal trees from the input tree sequence
    - "treespan": (l x 2) array storing the spans of genomic windows if `--window-size` is specified, should be the same as "marginal_treespan" otherwise
    - "marginal_mask": (l, ) shape 0-1 array specifiying if the marginal tree overlaps > 99% with specified `--include-regions`, 1-True, 0-False
    - "accessible_windows": (l, ) shape 0-1 array specifiying if the genomic window (if `--window-size` specified) contain any marginal trees that overlap > 99% with specified `--include-regions`, 1-True, 0-False; should be the same as "marginal_mask" otherwise
    - "individuals": (n, ) shape integer array specifiying the tree node ID of each sample record in "ncoal", "t1s", "t2s"

2. Output from `trace-infer`: a numpy NpzFile with following numpy arrays
    - Let l be the number of marginal trees (or genomic windows) for output chromosome i, m denote the number of posterior tree sequences (from SINGER) the user inputed (e.g. number of rows in each `--data-files`).
    - "ncoal": (l, ) shape array specifying final input observation data for TRACE (summarized across m posterior trees with `--func`)
    - "treespan": (l x 2) array storing the spans of marginal trees (or genomic windows) in genetic distance (cM), should be "treespan_phy"*1e-6 if `--genetic-maps` not specified
    - "treespan_phy": (l x 2) array storing the spans of marginal trees (or genomic windows) in physical distance (bp)
    - "accessible_windows": (l, ) shape 0-1 array specifiying if the marginal tree (or genomic window) is accessible in any one of the m posterior trees, 1-True, 0-False.
    - "params": (p x 9) array storing the inferred parameters at each EM updates (p rounds in total)
    - "gammas": (2 x l) array storing the posterior probability for 0-Human and 1-Archaic states across l marginal trees (or genomic windows)

3. Output from `trace-summarize`: a txt file with following columns
    - "chromosome": chromosome identifier, same as specified in `--chroms`
    - "start": start of the segment (in bp)
    - "end": end of the segment (in bp)
    - "mean_posterior": average posterior probabilities across marginal trees (or genomic windows) covered by the segment
    - "length(bp)": length of the segment (in bp)
    - "length(cM)": length of the segment (in cM)



## Contact

If interested in this work - please contact @aabiddanda or @YulinZhang9806 via a github issue.
