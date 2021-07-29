# PLEDGER
Pyopencl based tooL for gEnomic workloaDs tarGeting Embedded platfoRms

PLEDGER is an OpenCL based read mapper designed for embedded implementation. It is like an all-mapper and reports upto a specified number of mappings per read. It pre-processes the genome/genomic_section/chromosome using FM-Index and suffix array to produce the data structure files to be used while mapping reads. It, automatically, generates the folders and places various auxiliary files in them. It generates low-memory footprint data structures that can fit in embedded platforms with less than 3.6 GB of available RAM. It maps to each chromosome, sequentially, selects the chromosomes depending on user-given choice. It employs pigeonhole principle combined with dynamic programming based k-mer/seed selection criteria inspired from the Optimum Seed Solver[1]. Dynamic programming based filteration explores all possible lengths of k-mers, within given constraints, to reduce the total number of candidate locations. Using this filteration scheme it reduces the total number of verification steps significanlty, therefore, producing enhanced performance. 

PLEDGER can be used on any OpenCL conformant device. The host program is written in Python while the OpenCL kernel for parallel computation is written in C. We have implemented PLEDGER on different systems with different combinations, versions and technology of CPU, including Intel Core i5, i7, GPU, including Nvidia GTX490 and Tesla C1060, and embedded platforms such as Odroid N2. For detailed results please refer to the manuscript accepted in Design, Automation and Test in Europe Conference 2021 [2].  

PLEDGER has been divided into four parts: PLEDGER_preprocessing.py, PLEDGER_get_device_choice.py, PLEDGER_host.py and PLEDGER_kernel.cl

I) PLEDGER_preprocessing.py -- Usage instructions can be requested using the following:
   
    python3 PLEDGER_preprocessing.py -h 
    python3 PLEDGER_preprocessing.py Genome/chr21.fa chr21
   
The input accepts a fasta file and produces three data structure files in different folders viz. Tally_LMF, SA (suffix array), F_FMIndex, Tally_Offset and RefGenUppercase, where number encoded reference genome is stored. The overall memory required for the detastructure depends on the size of the chromosome or genomic section. We use online available programs with reusablity license to obtain suffix array of any genome. Source code and License files can be found in the "Build_SA.zip" folder. User may need to remake the project in their workspace and replace the executable file "suftest" with the latest one in the main working folder.

II) PLEDGER_get_device_choice.py -- Usage:

    python3 PLEDGER_get_device_choice.py 
    
    Example output:
    Following is the list of platforms:
      PLATFORM NO.    : PLATFORM NAME/S

       0        : [<pyopencl.Device 'Intel(R) Core(TM) i7-2600 CPU @ 3.40GHz' on 'Intel(R) OpenCL' at 0x270b348>] 

       1        : [<pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x27212a0>, <pyopencl.Device 'Tesla C1060' on 'NVIDIA CUDA' at 0x275a060>, <pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x2254df0>] 

      ----------------------------------------------------------------------

      Following is the list of all devices:
      DEVICE NO.      : DEVICE NAME

             0        : <pyopencl.Device 'Intel(R) Core(TM) i7-2600 CPU @ 3.40GHz' on 'Intel(R) OpenCL' at 0x270b348> 

             1        : <pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x27212a0> 

             2        : <pyopencl.Device 'Tesla C1060' on 'NVIDIA CUDA' at 0x275a060> 

             3        : <pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x2254df0> 

      ----------------------------------------------------------------------
    
The example above shows that the system under consideration has 2 platforms viz. Intel and Nvidia. And four devices altogether viz. two GTX590, one C1060 and one Intel quad-core i7 CPU. The device choices are required to be used when running the host program. This indicates on which devices does the user want to map reads. Note that installation of OpenCL SDK or drivers are required to obtain device choices. It, also, confirms the correct installation of the drivers. 

III) PLEDGER_host.py- Usage:

	python3 PLEDGER_host.py -h
	-----------------------------------------------------------------------
	usage: PLEDGER_host.py [-h] [-dc DC [DC ...]] [-rs RS [RS ...]] [-nr NR]
                       [-chr CHR [CHR ...]]
                       Fastq_filename Read_length Error Outputs klen

	PLEDGER: Pyopencl based tooL for gEnomic workloaDs tarGeting Embedded platfoRms

	positional arguments:
	  Fastq_filename      Give the name of reads file in fastq format.
	  Read_length         Provide the length of the reads. Choices: 100 or 150
	  Error               Provide the maximum permissible error. Smaller the value
	                      faster will be the algorithm. Range:[0-8]
	  Outputs             Maximum number of mappings allowed per read.
	                      Range:[1:3500]
	  klen                K-mer length. Range:[12:25]

	optional arguments:
	  -h, --help          show this help message and exit
	  -dc DC [DC ...]     Provide the numbers of all the devices to be used. These
	                      numbers can be from running 'get_device_choice.py'
	                      script
	  -rs RS [RS ...]     Provide the ratio for share of reads to be mapped for
	                      each device e.g. [0.6 0.2 0.2] for three devices. Sum of
	                      shares MUST be ONE.
	  -nr NR              Number of reads to be mapped, default: all the reads in
	                      the fastq file
	  -chr CHR [CHR ...]  Provide the serial numbers of chromosomes you want to
	                      map e.g. 1 2 21 would mean you want to map reads to
	                      chromosomes chr1, chr2 and chr21. To map to all the
	                      chromosomes, that were preprocessed, mention 'all'. The
	                      chromosomes include 1-22, X, Y


The device choices obtained with the 'PLEDGER_get_device_choice.py' script is used to specify the devices to be using for read mapping. Along with the devices, user is needed to specify the workload, i.e. number of reads, to be processed by each device chosen. 

      Example:
      python3 PLEDGER_host.py ERR012100_1_1000000_reads.fq 100 5 100 14 -dc 0 1 3 -rs 488000 256000 256000
      
      Here, read length is 100, permissible number of errors, i.e. edit distance, is 5, minimum k-mer length is 14,
      mapping locations per read is 100, chosen devices are the Intel quad-core CPU and two Nvidia GTX 590s. 
      The read distribution is 488000 to CPU and 256000 each to the two Nvidia GPUs.

IV) PLEDGER_kernel.cl : The main OpenCL kernel file which performs two functions: number encoding of reads and read mapping using filtration and verifications steps for all reads.
      
Prerequisites: Installation of OpenCL SDK and drivers for all the platforms available on the system. Python and PyOpenCL are also required to be installed on the systems. 

Note: Any problem while running PLEDGER will mainly be due to incorrect installations or memory allocation based issue. There should be sufficient memory to allocate for the data structure. OpenCL permits a maximum of 1/4th of the total RAM available on the device to any single variable. Thus, in case of large chromosomes such as chr2, it is advised to have atleast 16GB of RAM installed on the device. This limitation will be removed from future versions of the tool. If suftest executable throws error, we will recommend extracting the SA.zip and recompiling it. Then overwrite the old executable with the new one. 


[1] H. Xin, S. Nahar, R. Zhu, J. Emmons, G. Pekhimenko, C. Kingsford, C. Alkan, and O. Mutlu, “Optimal seed solver: optimizing seed selection in read mapping,” Bioinformatics, vol. 32, no. 11, pp. 1632–1642, 2016.

[2] S. Maheshwari, R. Shafik, I. Wilson, A. Yakovlev, V. Y. Gudur and A. Acharyya, "PLEDGER: Embedded Whole Genome Read Mapping using Algorithm-HW Co-design and Memory-aware Implementation," 2021 Design, Automation & Test in Europe Conference & Exhibition (DATE), 2021, pp. 1855-1858, doi: 10.23919/DATE51398.2021.9473909.

[3] S. Maheshwari, V. Y. Gudur, R. Shafik, I. Wilson, A. Yakovlev, and A. Acharyya, “Coral: Verification-aware opencl based read mapper for heterogeneous systems,” IEEE/ACM Transactions on Computational Biology and Bioinformatics, pp. 1–1, 2019.