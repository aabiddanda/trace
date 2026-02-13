# TRACE: Estimating Introgression using ancestral recombination graphs

## Installation

In order to install the package locally you can run:

```
git clone https://github.com/aabiddanda/trace.git
cd trace; pip install .
```

## Command-line Interface
```
> trace-extract --help
Usage: trace-extract [OPTIONS]

  TRACE-Extract CLI.

Options:
  -f, --tree-file PATH       Input data in tskit or tsz format.  [required]
  -t, --t-archaic FLOAT      Focal time for branch.  [required]
  -i, --individuals TEXT     List of sampled haplotypes to run analysis for,
                             comma separated (no spaces). Recognizes tree node
                             IDs (int).  [required]
  -w, --window-size INTEGER  Window size summarizing tree sequences (required
                             if working with multiple posterior tree sequences
                             like outputs from SINGER). If not provided, uses
                             the marginal trees directly.
  --chrom TEXT               chromosome ID for the tree sequence, must match
                             the chromosome ID in the include regions file.
  --include-regions PATH     A BED file containing the INCLUDE regions for the
                             tree sequence.
  -o, --out TEXT             Output file prefixes.  [default: trace; required]
  --help                     Show this message and exit.

> trace-infer --help  
Usage: trace-infer [OPTIONS]

  TRACE-Inference CLI.

Options:
  -i, --individual TEXT  the focal individual tree node id to run the HMM on,
                         only take a sample name if --sample-names is
                         specified.  [required]
  --npz-files TEXT       Input data in npz format (output from trace-extract).
                         If multiple chromosomes are provided, separate by
                         comma (no spaces).
  --data-files TEXT      a plain text file containing paths to .npz files (outputs
                         from trace-extract), one .npz file per line. If
                         multiple chromosomes are provided, provide one data
                         file per chromosome, separated by comma (no spaces).
  --chroms TEXT          chromosome ID for the tree sequence, must match the
                         chromosome ID in the include regions file. If
                         multiple chromosomes are provided, separate by comma
                         (no spaces).  [default: chr1]
  --func [mean|median]   Summarize function for windows across posterior tree
                         sequences.  [default: mean]
  --genetic-maps TEXT    a HapMap formatted genetic map (see https://ftp.ncbi.
                         nlm.nih.gov/hapmap/recombination/2011-01_phaseII_B37/
                         for hg19 HapMap genetic map),the 2nd and 4th column
                         (1-index) should be position (bp) and genetic
                         distance (cM); if multiple chromosomes are provided,
                         separate by comma (no spaces). assume a uniform
                         recombination rate of 1e-8 per bp per generation if
                         not specified
  --seed INTEGER         random seed   [default: 42]
  -o, --out TEXT         Output file prefix, output files will be named as
                         [out].[chrom].xss.npz  [default: trace; required]
  --help                 Show this message and exit.

> trace-summarize --help
Usage: trace-summarize [OPTIONS]

  TRACE-Summarize CLI.

Options:
  -f, --files TEXT                Posterior probability file from trace-infer,
                                  end with .xss.npz. Multiple files (for the
                                  same individual, different chromosomes) are
                                  allowed, separated by comma  [required]
  -c, --chroms TEXT               Chromosome ID used in the output file, must
                                  be consistent with the input files
                                  [required]
  --posterior-threshold FLOAT     posterior probability threshold for calling
                                  introgression  [default: 0.9]
  --physical-length-threshold INTEGER
                                  physical length threshold for calling
                                  introgression, in bp  [default: 50000]
  --genetic-distance-threshold FLOAT
                                  genetic distance threshold for calling
                                  introgression, in cM  [default: 0.05]
  -o, --out TEXT                  prefix for output file, output file will be
                                  named as [out].summary.txt  [required]
  --help                          Show this message and exit.
```
## Running an Example

In order to facilitate a basic example, we have provided some simulated data to get a feel for using the CLI.


1. To extract relevant observation data for the HMM, we use:

```
trace-extract -f example_data/test_input.tsz -t 15000 -i 0,1,2,3,4,5,6,7,8,9,10 -o example_data/test_extract
```

2. To infer a posterior decoding of introgression tracts, we use:

```
trace-infer -i 2 --npz-files example_data/test_extract.npz -o example_data/test_infer
```

3. To get final archaic introgression tracts, we use:

```
trace-summarize -f example_data/test_infer.chr1.xss.npz -c chr1 -o example_data/test_summarize
```

## Example with Relate and SINGER inferred ARGs

Here we want to show an example of applying TRACE to ARGs inferred from real data. We would assume we are studying Neanderthal introgression into modern humans, so would use `--t-archaic 15000` (the user-defined timescale parater t=15000) for our analysis. This parameter should be chosen based on the aim of the study.

### Handling Relate and SINGER outputs

I would assume that we already have Relate or SINGER run on a dataset that we are interested in analyzing here. Since TRACE only accepts tree sequences (`.trees` or `.tsz`) as input, we need to convert raw outputs from these programs to `tskit` formats.

For Relate, please checkout https://github.com/leospeidel/relate_lib.

For SINGER, please checkout https://github.com/popgenmethods/SINGER for module `convert_to_tskit`

### Extracting observation data from ARGs

First we need to extract observation data for individuals that we are analyzing from ARGs. For this example, we would analyze haplotypes that has sample ID 0 to 3 (individual 0 to 1) in the input ARGs. Here are sample information for the 4 haplotypes

```
tree_node_id    haplotype_name
0   ind0_hap1
1   ind0_hap2
2   ind1_hap1
3   ind2_hap2
```

Relate produces one tree sequence per chromosome. Example output files for a dataset (name: dataset1) would have a structure like this

```
relate
├── dataset1_chr1.tsz
├── dataset1_chr2.tsz
└── dataset1_chr3.tsz
```

SINGER usually outputs multiple trees per chromosome depending on the input parameters we set. I would assume that we sampled 3 posterior tree sequences per chromosome when running SINGER. Then an example result directory would have a structure like this

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

We need to extract observation data for haplotype 0-3 (sample node ID 0-3, individual 0-1) from all tree sequences (`.tsz` files) provided. An example run on one of the Relate output tree sequence would be

```
# This would produce output file relate/dataset1_t15000_group1_chr1.npz
> trace-extract --tree-file relate/dataset1_chr1.tsz --t-archaic 15000 --individuals 0,1,2,3 -o relate/dataset1_t15000_group1_chr1
```

We could ask TRACE to only use genotype information from regions with high confidence (for example, strict / pilot masks from 1000 Genomes) by specifying `--include-regions` and `--chrom`. This would limit the following analysis on trees that overlap >99% with the input BED file in the tree sequence.

```
# This would produce output file relate/dataset1_t15000_strictmask_group1_chr1.npz
> trace-extract --tree-file relate/dataset1_chr1.tsz --t-archaic 15000 --individuals 0,1,2,3 --include-regions strictmask_chr1.bed --chrom chr1 -o relate/dataset1_t15000_strictmask_group1_chr1
```

For SINGER outputs, we need to specify `--window-size` parameter so that TRACE could summarize results across different posterior tree sequences.

```
# This would produce output file singer/chr1/dataset1_t15000_strictmask_group1_chr1_sample1.npz
> trace-extract --tree-file singer/chr1/dataset1_chr1_sample1.tsz --t-archaic 15000 --individuals 0,1,2,3 --include-regions strictmask_chr1.bed --chrom chr1 --window-size 1000 -o singer/chr1/dataset1_t15000_strictmask_group1_chr1_sample1
```

We need to run this command separately for each tree sequence file. We recommand extracting multiple samples in one command, which would make the most efficient usage of computation time, memory and storage space. However, this step does take some amount of time when the input chromosome is large. In this case, splitting individuals into groups and running different groups in parallel would be the best choice.

### Running TRACE inference

After the previous step, our result folders should now contain `.npz` files extracted from each `.tsz` file. An example directory structure would look like (`.tsz` files are not shown since they are not relevant)

```
relate
├── dataset1_t15000_strictmask_group1_chr1.npz
├── dataset1_t15000_strictmask_group1_chr2.npz
└── dataset1_t15000_strictmask_group1_chr3.npz

singer
├── chr1
│   ├── dataset1_t15000_strictmask_group1_chr1_sample1.npz
│   ├── dataset1_t15000_strictmask_group1_chr1_sample2.npz
│   └── dataset1_t15000_strictmask_group1_chr1_sample3.npz
├── chr2
│   ├── dataset1_t15000_strictmask_group1_chr2_sample1.npz
│   ├── dataset1_t15000_strictmask_group1_chr2_sample2.npz
│   └── dataset1_t15000_strictmask_group1_chr2_sample3.npz
└── chr3
    ├── dataset1_t15000_strictmask_group1_chr3_sample1.npz
    ├── dataset1_t15000_strictmask_group1_chr3_sample2.npz
    └── dataset1_t15000_strictmask_group1_chr3_sample3.npz
```

We then need to run TRACE on each haplotype separately. An example run on haplotype 2 on Relate outputs applying genetic maps from HapMap would be

```
# define inputs
> npzfile_string=$(printf "relate/dataset1_t15000_strictmask_group1_chr%d.npz," {1..3} | sed 's/,$//')
> chrom_string=$(printf "chr%d," {1..3} | sed 's/,$//')
> gmap_string=$(printf "genetic_map_hg38_chr%d.txt," {1..3} | sed 's/,$//')

# This would produce three output files: relate/dataset1_t15000_strictmask_ind2.chr1.xss.npz, relate/dataset1_t15000_strictmask_ind2.chr2.xss.npz, relate/dataset1_t15000_strictmask_ind2.chr3.xss.npz
> trace-infer -i 2 --npz-files $npzfile_string --chroms $chrom_string --genetic-maps $gmap_string -o relate/dataset1_t15000_strictmask_ind2
```

To run TRACE on SINGER outputs, we need to first prepare data files containing all SINGER posterior tree sequences information for the same chromosome. An example for chromosome 1 in this example would be

```
# Same files could be produced for chr2 and chr3
> cat singer_data_chr1.txt
singer/chr1/dataset1_t15000_strictmask_group1_chr1_sample1.npz
singer/chr1/dataset1_t15000_strictmask_group1_chr1_sample2.npz
singer/chr1/dataset1_t15000_strictmask_group1_chr1_sample3.npz
```

Then, we need to run TRACE on each haplotype separately. An example run on haplotype 2 applying genetic maps from HapMap would be (note we use `--data-files` instead of `--npz-files` for inputs)

```
# define inputs
> datafile_string=$(printf "singer/singer_data_chr%d.txt," {1..3} | sed 's/,$//')
> chrom_string=$(printf "chr%d," {1..3} | sed 's/,$//')
> gmap_string=$(printf "genetic_map_hg38_chr%d.txt," {1..3} | sed 's/,$//')

# This would produce three output files: singer/dataset1_t15000_strictmask_ind2.chr1.xss.npz, singer/dataset1_t15000_strictmask_ind2.chr2.xss.npz, singer/dataset1_t15000_strictmask_ind2.chr3.xss.npz
> trace-infer -i 2 --data-files $datafile_string --chroms $chrom_string --genetic-maps $gmap_string -o singer/dataset1_t15000_strictmask_ind2
```

We need to run this command separately for each individual. We recommand analyzing all chromosomes together, as short chromosomes might have too limited data to train the Hidden Markov Model accurately. TRACE automatically outputs results for each chromosome separately. We recommand to parallelize this step across individuals.

### Getting archaic segments

Finally, we run `trace-summarize` to get archaic segments on each individual from TRACE inference results. From the previous step, our result folders should now contain `.xss.npz` files for each individual haplotype for every chromosome.

```
.
├── dataset1_t15000_strictmask_ind0.chr1.xss.npz
├── dataset1_t15000_strictmask_ind0.chr2.xss.npz
├── dataset1_t15000_strictmask_ind0.chr3.xss.npz
├── dataset1_t15000_strictmask_ind1.chr1.xss.npz
├── dataset1_t15000_strictmask_ind1.chr2.xss.npz
├── dataset1_t15000_strictmask_ind1.chr3.xss.npz
├── dataset1_t15000_strictmask_ind2.chr1.xss.npz
├── dataset1_t15000_strictmask_ind2.chr2.xss.npz
├── dataset1_t15000_strictmask_ind2.chr3.xss.npz
├── dataset1_t15000_strictmask_ind3.chr1.xss.npz
├── dataset1_t15000_strictmask_ind3.chr2.xss.npz
└── dataset1_t15000_strictmask_ind3.chr3.xss.npz
```

An example of getting archaic segments for haplotype 0 from results above is

```
# define inputs
> summarizefile_string=$(printf "dataset1_t15000_strictmask_ind0.chr%d.xss.npz," {1..3} | sed 's/,$//')
> chrom_string=$(printf "chr%d," {1..3} | sed 's/,$//')

# This would produce output file: dataset1_t15000_strictmask_ind2.summary.txt
> trace-summarize -f $summarizefile_string -c $chrom_string -o dataset1_t15000_strictmask_ind2
```

We need to run this command separately for each individual. We could change the filters applied to TRACE outputs by specifying `--posterior-threshold`, `--physical-length-threshold` and `--genetic-distance-threshold`.

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
    - Let l be the number of marginal trees (or genomic windows) for an input chromosome, m denote the number of posterior tree sequences (from SINGER) the user inputed (e.g. number of rows in each `--data-files`).
    - "ncoal": (l, ) shape array specifying final input observation data for TRACE (summarized across m posterior trees with `--func`)
    - "treespan": (l x 2) array storing the spans of marginal trees (or genomic windows) in genetic distance (cM), should be "treespan_phy"*1e-6 if `--genetic-maps` not specified
    - "treespan_phy": (l x 2) array storing the spans of marginal trees (or genomic windows) in physical distance (bp)
    - "accessible_windows": (l, ) shape 0-1 array specifiying if the marginal tree (or genomic window) is accessible in any one of the m posterior trees, 1-True, 0-False.
    - "params": (p x 9) array storing the inferred parameters at each EM updates (p rounds in total)
    - "gammas": (2 x l) array storing the posterior probability for row0-Human and row1-Archaic states across l marginal trees (or genomic windows)
    - "seed": (1, ) array storing the random seed used for TRACE run
    - "individual": (1, ) array storing tree node ID for the focal individual

3. Output from `trace-summarize`: a txt file **containing inferred archaic fragments** with following columns
    - "chromosome": chromosome identifier, same as specified in `--chroms`
    - "start": start of the segment (in bp)
    - "end": end of the segment (in bp)
    - "mean_posterior": average posterior probabilities across marginal trees (or genomic windows) covered by the segment
    - "length(bp)": length of the segment (in bp)
    - "length(cM)": length of the segment (in cM)


## Running Unit and Integration Testing (Developer Note)

We have implemented a testing suite using `pytest` which addresses much of the functionality of the main repository.

```
pip install .[tests] # install extra dependencies just for testing
pytest --verbose -n <cores>
```

## Contact

For questions/bug reports, please contact @aabiddanda or @YulinZhang9806. For identified bugs, please submit an issue with a minimally working example.
