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

In order to facilitate a basic example, we have provided some simulated data to get a feel for using the CLI.


1. To extract relevant observation data for the HMM, we use:

```
trace-extract --tree-file example_data/n10_seed10_A.tsz --t-archaic 15000 -s 0,1,2,3,4,5,6,7,8,9,10 -o example_data/test_output
```

2. To infer a posterior decoding of introgression tracts, we use:

```
trace-infer --individual 2 --npz-files example_data/test_output.npz -o example_data/test_infer
```

3. To get final archaic introgression tracts, we use:

```
trace-summarize --files example_data/test_infer.chr1.xss.npz --chroms chr1 --out example_data/test_summarize
```

## Example with Relate and SINGER inferred ARGs

Here I want to show an example of applying TRACE to ARGs inferred from real data. I would assume we are studying Neanderthal introgression into modern humans, so would use `--t-archaic 15000` (the user-defined timescale parater t=15000) for our analysis. This parameter should be chosen based on the aim of the study.

### Handling Relate and SINGER outputs

I would assume that we already have `Relate` or `SINGER` run on a dataset that we are interested in analyzing here. Since TRACE only accepts tree sequences (`.trees` or `.tsz`) as input, we need to convert raw outputs from these programs to `tskit` formats.

For `Relate`, please checkout https://github.com/leospeidel/relate_lib.

For `SINGER`, please checkout https://github.com/popgenmethods/SINGER for module `convert_to_tskit`

### Extracting observation data from ARGs

First we need to extract observation data for individuals that we are analyzing from ARGs. For this example, I would analyze haplotypes that has sample ID 0 to 3 (individual 0 to 1) in the input ARGs. Here is a file (dataset1_samples.txt) containing sample information (which could serve as the input for `--sample-names`)

```
> cat dataset1_samples.txt
0   ind0_hap1
1   ind0_hap2
2   ind1_hap1
3   ind2_hap2
```

`Relate` produces 1 tree sequence per chromosome. Example output files for a dataset (name: dataset1) would have a structure like this

```
relate
├── dataset1_chr1.tsz
├── dataset1_chr2.tsz
└── dataset1_chr3.tsz
```

`SINGER` usually outputs multiple trees per chromosome depending on the input parameters we set. I would assume that we sampled 3 posterior tree sequences per chromosome when running `SINGER`. Then an example result directory would have a structure like this

```
singer
├── chr1
│   ├── dataset1_chr1_sample1.tsz
│   ├── dataset1_chr1_sample2.tsz
│   └── dataset1_chr1_sample3.tsz
├── chr2
│   ├── dataset1_chr2_sample1.tsz
│   ├── dataset1_chr2_sample2.tsz
│   └── dataset1_chr2_sample3.tsz
└── chr3
    ├── dataset1_chr3_sample1.tsz
    ├── dataset1_chr3_sample2.tsz
    └── dataset1_chr3_sample3.tsz
```

We need to extract observation data for haplotype 0-3 (sample node ID 0-3, individual 0-1) from all tree sequences provided. An example run on one of the `Relate` output tree sequence would be

```
# This would produce output file relate/dataset1_t15000_group1_chr1.npz
> trace-extract --tree-file relate/dataset1_chr1.tsz --t-archaic 15000 --samples 0,1,2,3 -o relate/dataset1_t15000_group1_chr1
```

By specifying `--sample-names`, we could use self-defined names to specify these samples (check dataset1_samples.txt file we showed earlier)

```
# This would produce output file relate/dataset1_t15000_group1_chr1.npz
> trace-extract --tree-file relate/dataset1_chr1.tsz --t-archaic 15000 --samples ind0_hap1,ind0_hap2,ind1_hap1,ind1_hap2 --sample-names dataset1_samples.txt -o relate/dataset1_t15000_group1_chr1
```

We could ask TRACE to only use genotype information from regions with high confidence (for example, strict / pilot masks from 1000 Genomes) by specifying `--include-regions` and `--chrom`. This would limit the following analysis on trees that overlap >99% with the input BED file in the tree sequence.

```
# This would produce output file relate/dataset1_t15000_strictmask_group1_chr1.npz
> trace-extract --tree-file relate/dataset1_chr1.tsz --t-archaic 15000 --samples 0,1,2,3 --include-regions strictmask_chr1.bed --chrom chr1 -o relate/dataset1_t15000_strictmask_group1_chr1
```

For `SINGER` outputs, we need to specify `--window-size` parameter so that TRACE could summarize results across different posterior tree sequences.

```
# This would produce output file singer/chr1/dataset1_t15000_strictmask_group1_chr1_sample1.npz
> trace-extract --tree-file singer/chr1/dataset1_chr1_sample1.tsz --t-archaic 15000 --samples 0,1,2,3 --include-regions strictmask_chr1.bed --chrom chr1 --window-size 1000 -o singer/chr1/dataset1_t15000_strictmask_group1_chr1_sample1
```

We need to run this command separately for each tree sequence file. We recommand extracting multiple samples in one command, which would make the most efficient usage of computation time, memory and storage space. However, this step does take some amount of time when the input chromosome is large. In this case, splitting individuals into groups and running different groups in parallel would be the best choice.

### Running TRACE inference


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
    - "seed": (1, ) array storing the random seed used for TRACE run
    - "individual": (1, ) array storing tree node ID for the focal individual

3. Output from `trace-summarize`: a txt file with following columns
    - "chromosome": chromosome identifier, same as specified in `--chroms`
    - "start": start of the segment (in bp)
    - "end": end of the segment (in bp)
    - "mean_posterior": average posterior probabilities across marginal trees (or genomic windows) covered by the segment
    - "length(bp)": length of the segment (in bp)
    - "length(cM)": length of the segment (in cM)


## Contact

If interested in this work - please contact @aabiddanda or @YulinZhang9806 via a Github issue.
