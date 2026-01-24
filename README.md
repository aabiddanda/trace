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


1. To extract relevant emissions for the HMM, we use:

```
trace-extract --tree-file example_data/n10_seed8_A.tsz -s 0,1,2,3,4,5,6,7,8,9,10 -o example_data/test_output
```

2. To infer a posterior decoding of introgression tracts, we use: 

```
trace-infer --individual 2 --npz-file example_data/test_output.npz -o example_data/test_infer
```

3. 


## Contact

If interested in this work - please contact @aabiddanda or @YulinZhang9806 via a github issue.
