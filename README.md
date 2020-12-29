# Accelerating Adaptive Banded Event Alignment Algorithm on FPGAs using OpenCL

## Aim of the proposed project:
- Effective utilization of OpenCL to map the Adaptive Banded Event Alignment(ABEA) algorithm to run efficiently on an FPGA. 
- Evaluate the performance improvement of the ABEA with the FPGA implementation.

## Background
The process of DNA sequencing is a precise determination of the amount and distribution of nucleotides (adenine (A), guanine (G), cytosine (C), and thymine (T)) in DNA molecules. It has a very strong impact in various biological fields such as human genetics, agriculture, bioinformatics, etc.

DNA sequencing machines produce gene sequences much faster than the traditional molecular biology techniques and also these DNA sequencing data is much larger in size (terabytes of data, read lengths of 1000 to >1M bases).  Analyzing these data still depends on high-performance or cloud computers. 

Therefore, accelerating DNA sequencing methods by heterogeneous architectures ( i.e.FPGAs) and the capability of detecting entire genomes in short periods of time could revolutionize the world of medicine and technology.

Therefore, to overcome this immense computational load, reconfigurable computing, also known as FPGA is the field in which algorithms are mapped directly to configurable hardware resources with parallelism. The cost per computation and watts per computation is also quite favorable hence it is worth running the bioinformatics algorithms on FPGAs.

In the field of Nanopore sequencing, the ABEA algorithm can be used to align the raw signal (a time series of electric current that can be generated using the latest generation (third generation) of sequencing technologies) to a biological reference sequence.

ABEA is one of the most time-consuming steps when analyzing raw nanopore data. As of now it has been parallelized, optimized, and fine-tuned to exploit architectural features in general-purpose GPUs and CPUs, and in this project, we propose to take these techniques to FPGAs.

A custom hardware design of the ABEA algorithm done with hardware-software co-design principles has the potential to achieve superior performance.OpenCL can be used for writing programs at high-level languages on FPGA which are then converted by the underlying layers to run with the support of board support package (BSP) in order to accelerate the ABEA Algorithm on it.

## Usage

#### Create the Dataset
```

```

#### Compile individual kernel for de5net
```
./scripts/compile_kernel_de5net pre bins/bin
```

#### Compile all kernels for de5net
```
./scripts/compile_all_kernels_de5net bins/bin
```

#### Compile and run host code for small dataset
```
make BIN=bins/bin
./bins/bin/host ../dump_small
```
