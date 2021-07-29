import numpy as np
import pyopencl as cl
from pyopencl import array
import sys
import subprocess
import time
import ast
import collections
from itertools import islice
import pickle
import math
import statistics
import os
import argparse

# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
def HCF(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i
            
    return hcf

def print_platforms_and_devices():
	all_platforms = dict()
	platforms = cl.get_platforms()
	num_plts = 0
	for pltm in platforms:
		if(pltm.name[:12] != 'Experimental'):	
			if(pltm.get_devices(cl.device_type.CPU) != []):
				all_platforms[num_plts] = ["CPU", pltm.get_devices(cl.device_type.CPU), cl.Context(devices=pltm.get_devices(cl.device_type.CPU)),pltm]
				num_plts = num_plts + 1
				# all_platforms.append(pltm.get_devices(cl.device_type.CPU))

	for pltm in platforms:
		if(pltm.name[:12] != 'Experimental'):	
			if(pltm.get_devices(cl.device_type.GPU) != []):
				all_platforms[num_plts] = ["GPU",pltm.get_devices(cl.device_type.GPU), cl.Context(devices=pltm.get_devices(cl.device_type.GPU)),pltm]
				num_plts = num_plts + 1
	num_devices = 0
	all_devices = dict()
	for key,value in all_platforms.items():
		for i in value[1]:
			all_devices[num_devices] = [value[0],value[2], i, value[3]] ## Device type, Context, Device name
			num_devices = num_devices + 1
	return all_devices  # returning CPU context and CPU device type separately to be used in preprocessing reads


def Write_Output(cand_locs_per_read, strand, mapped_endpos_read, device_choices, sequences, sequence_names, no_of_outputs_per_read, filename, n, p, chr_name):
	heading = 'First column: Read name; 	Second column: Strand;		Third column: End position in the genome where read matched;	Fourth column: Edit distance; 		Fifth columbn: Read sequence\n'
	arr = []
	read_number = 0
	with open(p + 'PLEDGER_Output/' + filename[:5]+ '_'+ chr_name +'.repute','w') as f:
		f.write(heading)
		for i in device_choices:
			temp_list = list(cand_locs_per_read[i])					
			for j in range(len(temp_list)):		
				for k in range(temp_list[j]):
					if(strand[i][j*no_of_outputs_per_read + k] >= 128):
						strnd = 'F'
						ed = strand[i][j*no_of_outputs_per_read + k] - 128
					else:
						strnd = 'R'
						ed = strand[i][j*no_of_outputs_per_read + k]
					if(k == 0):
						arr = [sequence_names[read_number][1:], strnd, str(mapped_endpos_read[i][j*no_of_outputs_per_read + k]), str(ed), str(sequences[read_number])]#, str(n)
					else:
						arr = [sequence_names[read_number][1:], strnd, str(mapped_endpos_read[i][j*no_of_outputs_per_read + k]), str(ed), "\""]#, str(n)
					# f.write("{:25}{:^10}{:^22}{:^8}{:^10}{:^150}".format(*arr)+'\n')
					f.write('\t'.join(arr[0:]) + '\n')
				read_number += 1




def opencl(n, e, device_choices, no_of_reads, no_of_outputs_per_read, share_ratio_per_device, q_len, chromosome_list, p, sequence_names, filename, sequences):	
	no_of_reads = np.uint32(no_of_reads)
	device_list = print_platforms_and_devices()
	n = np.uint32(n)
	#---------------------------------------------------------------------------------------------------------
	reads = ''.join(sequences)
	reads = np.array(reads,'c')
	command_queue = cl.CommandQueue(device_list[0][1], device_list[0][2], cl.command_queue_properties.PROFILING_ENABLE)
	#preprocessing reads only on CPU
	r_fwd, r_rev = process_reads(no_of_reads, reads, n, device_list[0][1], command_queue)

	del reads
	#---------------------------------------------------------------------------------------------------------
	l_dev_choice = len(device_choices)
	share_per_device = [0]*l_dev_choice
	no_of_reads_remaining = no_of_reads
	reads_allocated = 0
	tmp = 0
	reads_fwd = dict()
	reads_rev = dict()
	for i in range(l_dev_choice):
		if(i+1 < l_dev_choice):
			tmp = int((((share_ratio_per_device[i]*no_of_reads)//32)+1)*32)
			if(tmp > no_of_reads):
				share_per_device[i] = no_of_reads_remaining
			else:
				share_per_device[i] = tmp
			no_of_reads_remaining = no_of_reads_remaining - share_per_device[i]			
		else:
			share_per_device[i] = no_of_reads_remaining
		reads_fwd[device_choices[i]] = r_fwd[reads_allocated*n:(reads_allocated + share_per_device[i])*n]
		reads_rev[device_choices[i]] = r_rev[reads_allocated*n:(reads_allocated + share_per_device[i])*n]
		reads_allocated = reads_allocated + share_per_device[i]
	# print(share_per_device, reads_fwd, reads_rev)

	del tmp; del l_dev_choice; del no_of_reads_remaining;del reads_allocated;
	# print('{:<30}'.format('No. of valid reads to mapped'),':',no_of_reads)
	for chr_name in chromosome_list:
		print('{:<30}'.format('Processing chromosome'),':',chr_name)
		#---Read integer coded genome file 
		genome = np.load(p + 'RefGenUppercase/' + chr_name +'_genome_uppercase.npy', allow_pickle=False)
		# print(genome[:100])
		#---Read suffix array
		SA =  np.load(p + 'SA/' + chr_name +'_SA.npy', allow_pickle=False)
		# print(SA[:100])
		#---Read Tally matrix of the FM Index
		tally = np.load(p + 'Tally_LMF/' + chr_name + '_Tally_LMF.npy', allow_pickle=False)
		# print(tally[:100])
		#---Read F array of the FM Index giving cummulated count of the bases A,C,G and T
		F = np.load(p + 'F_FMIndex/' + chr_name + '_F_FMIndex.npy', allow_pickle=False)	
		# print(F)
		#---t_offset array
		t_offset = np.load(p + 'Tally_Offset/' + chr_name + '_Tally_offset.npy', allow_pickle=False)	
		# print(t_offset[:100])

		total_memory_required = SA.nbytes + tally.nbytes + F.nbytes + sys.getsizeof(reads_fwd) + no_of_reads*4 + no_of_reads*no_of_outputs_per_read*5 + t_offset.nbytes + genome.nbytes
		print('{:<30}{:^3}{:.2f}'.format('Minimum memory required',':',total_memory_required/(1024*1024*1024)),'GB')
		chosen_device_parameters = dict()
		for key,value in device_list.items():
			if(value[2].global_mem_size < total_memory_required):
				print('Not enough RAM')	
			chosen_device_parameters[key] = [value[1], cl.CommandQueue(value[1], value[2], cl.command_queue_properties.PROFILING_ENABLE), value[3], value[2], value[0]]

		strand_of_read, mapped_endpos_read, cand_locs_per_read = launching_kernel(reads_fwd, reads_rev, genome, chosen_device_parameters, no_of_reads, n, no_of_outputs_per_read, e, tally, SA, F, share_per_device, device_choices,q_len, t_offset)
		print('Writing output to file')
		Write_Output(cand_locs_per_read, strand_of_read, mapped_endpos_read, device_choices, sequences, sequence_names, no_of_outputs_per_read, filename, n, p, chr_name)
		print('-----------------------------------------------------------------------')

def process_reads(l, reads, n, cpu_context, cpu_command_queue):
	program_file = open('PLEDGER_kernel.cl', 'r')
	program_text = program_file.read()
	program_file.close()
	program_compilation_options = ["-Werror","-cl-mad-enable","-cl-denorms-are-zero"]#,"-cl-opt-disable" ,"-cl-std=CL1.2" ,"-cl-uniform-work-group-size"
	temp_text = "-DRLEN=" + str(n)
	program_compilation_options.append(temp_text)
	temp_text = "-DWORD_LENGTH=" + str(32)
	program_compilation_options.append(temp_text)
	temp_text = "-DPERMISSIBLE_ERROR=" + str(5)
	program_compilation_options.append(temp_text)
	temp_text = "-DCANDIDATES_PER_READ=" + str(200)
	program_compilation_options.append(temp_text)
	temp_text = "-DUINT_WITH_MSB_ONE=" + str(2147483648)
	program_compilation_options.append(temp_text)
	temp_text = "-DMIN_QGRAM_LEN=" + str(12)  # minimum q-gram length - 1 (minus one), minus one just to prevent an extra computation
	program_compilation_options.append(temp_text)
	print('-----------------------------------------------------------------------')
	print('IGNORE THE FOLLOWING MESSAGE')
	program = cl.Program(cpu_context, program_text)	
	try:
		program.build(program_compilation_options)
	except:
		print('Problem in building the kernel')
		raise
	print('-----------------------------------------------------------------------')
	kernel = cl.Kernel(program, 'preprocess_read')
	Reads_fwd = np.zeros(l*n, np.uint8)
	Reads_rev_cmp = np.zeros(l*n, np.uint8)
	reads = cl.Buffer(cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=reads)
	kernel.set_arg(0, reads)
	buffer_Reads_fwd = cl.Buffer(cpu_context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=Reads_fwd)
	kernel.set_arg(1, buffer_Reads_fwd)
	buffer_Reads_rev_cmp = cl.Buffer(cpu_context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=Reads_rev_cmp)
	kernel.set_arg(2, buffer_Reads_rev_cmp)

	start_time = time.time()
	event = cl.enqueue_nd_range_kernel(cpu_command_queue, kernel, (l,) , (1,))
		# event[i] = cl.enqueue_task(chosen_device_parameters[i][1], kernel[i], wait_for=None)

	event.wait()
	total_time = time.time() - start_time
	print('{:<31}{:<2}{:.4f}'.format('Read preprocessing time',':',total_time),'s')
	return Reads_fwd, Reads_rev_cmp


def launching_kernel(reads_fwd, reads_rev, genome, chosen_device_parameters, no_of_reads, read_len, no_of_outputs_per_read, e, tally, SA, F, share_per_device, device_choices,q_len, t_offset):
	
	program_file = open('PLEDGER_kernel.cl', 'r')
	program_text = program_file.read()
	program_file.close()	
	no_of_outputs_per_read = np.uint32(no_of_outputs_per_read)
	word_size = 32
	q_gram_len = np.uint32(q_len)	
	print('{:<30}'.format('Genome length'),':', len(genome))
	print('{:<30}'.format('Total no of reads'),':',no_of_reads)
	print('{:<30}'.format('Minimum k-mer length'),':', q_gram_len)
	#-------------------------------------------------------------------------------------------------------
	#Setting kernel compilation parameters: passing constants to the kernel
	program_compilation_options = ["-Werror","-cl-mad-enable","-cl-denorms-are-zero"]#,"-cl-opt-disable" ,"-cl-std=CL1.2" ,"-cl-uniform-work-group-size"
	temp_text = "-DRLEN=" + str(read_len)
	program_compilation_options.append(temp_text)
	temp_text = "-DWORD_LENGTH=" + str(word_size)
	program_compilation_options.append(temp_text)
	temp_text = "-DPERMISSIBLE_ERROR=" + str(e)
	program_compilation_options.append(temp_text)
	temp_text = "-DCANDIDATES_PER_READ=" + str(no_of_outputs_per_read)
	program_compilation_options.append(temp_text)
	temp_text = "-DUINT_WITH_MSB_ONE=" + str(2147483648)
	program_compilation_options.append(temp_text)
	temp_text = "-DMIN_QGRAM_LEN=" + str(q_gram_len)  # minimum q-gram length - 1 (minus one), minus one just to prevent an extra computation
	program_compilation_options.append(temp_text)
	print('-----------------------------------------------------------------------')
	print('IGNORE THE FOLLOWING MESSAGE')
	num_reads_alloted = 0
	program = dict()
	kernel = dict()
	Genome = dict()
	Reads_fwd = dict()
	Reads_rev_cmp = dict()
	buffer_SA = dict()
	buffer_tally = dict()
	buffer_F = dict()
	buffer_t_offset = dict()
	cand_locs_per_read = dict()
	buffer_cand_locs_per_read = dict()
	strand_of_read = dict()
	buffer_strand_of_read = dict()
	mapped_endpos_read = dict()
	buffer_mapped_endpos_read = dict()
	num_work_items_per_device = dict()
	for i in device_choices:
		program[i] = cl.Program(chosen_device_parameters[i][0], program_text)
	# for i in program:
		try:
			program[i].build(program_compilation_options)  #
		except:
			print('Problem in building the kernel')
			# print("Build log:")
			# print(program[i].get_build_info(chosen_device_parameters[i][1],cl.program_build_info.LOG))
			raise
		print('-----------------------------------------------------------------------')
		kernel[i] = cl.Kernel(program[i], 'pledger')
		if(chosen_device_parameters[i][4] == "CPU"):
			num_work_items_per_device[i] = int(HCF(share_per_device[i], 64))
		elif(chosen_device_parameters[i][4] == "GPU"):
			num_work_items_per_device[i] = int(HCF(share_per_device[i], 8))

		print('{:<25}'.format('Device selected'),':',chosen_device_parameters[i][3].name)
		print('{:<26}{:<2}{:.1f}'.format('Global MEM size (MB)',':',(chosen_device_parameters[i][3].global_mem_size)/(1024*1024)), 'MB')
		print('{:<25}'.format('Local MEM size (Bytes)'),':',(chosen_device_parameters[i][3].local_mem_size))
		print('{:<25}'.format('Max work group size'),':',(chosen_device_parameters[i][3].max_work_group_size))
		print('{:<25}'.format('Max work item size'),':',(chosen_device_parameters[i][3].max_work_item_sizes),'\n')
		print('{:<75}'.format('KERNEL USAGE INFORMATION'))
		print('{:<75}'.format('Kernel Name'),':',kernel[i].get_info(cl.kernel_info.FUNCTION_NAME))
		print('{:<75}'.format('Maximum Work group size'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE,chosen_device_parameters[i][3]))
		print('{:<75}'.format('A multiple for determining work-group sizes that ensure best performance'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,chosen_device_parameters[i][3]))
		print('{:<75}'.format('Local memory used by the kernel in bytes'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.LOCAL_MEM_SIZE,chosen_device_parameters[i][3]))
		print('{:<75}'.format('Private memory used by the kernel in bytes'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.PRIVATE_MEM_SIZE,chosen_device_parameters[i][3]))
		print('{:<75}'.format('Work items per work group'),':',num_work_items_per_device[i],'\n')

		if(chosen_device_parameters[i][4] == "CPU"):# That means its CPU, so the flags will be different
			Genome[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=genome)
			kernel[i].set_arg(0, Genome[i])
			Reads_fwd[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=reads_fwd[i])
			kernel[i].set_arg(1, Reads_fwd[i])
			Reads_rev_cmp[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=reads_rev[i])
			kernel[i].set_arg(2, Reads_rev_cmp[i])
			buffer_SA[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=SA)
			kernel[i].set_arg(3, buffer_SA[i])
			buffer_tally[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=tally)
			kernel[i].set_arg(4, buffer_tally[i])
			buffer_F[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=F)
			kernel[i].set_arg(5, buffer_F[i])
			cand_locs_per_read[i] = np.zeros(share_per_device[i], np.uint32)
			buffer_cand_locs_per_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=cand_locs_per_read[i])
			kernel[i].set_arg(6, buffer_cand_locs_per_read[i])
			strand_of_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint8)
			buffer_strand_of_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=strand_of_read[i])
			kernel[i].set_arg(7, buffer_strand_of_read[i])
			mapped_endpos_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint32)
			buffer_mapped_endpos_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=mapped_endpos_read[i])
			kernel[i].set_arg(8, buffer_mapped_endpos_read[i])		
			buffer_t_offset[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=t_offset)
			kernel[i].set_arg(9, buffer_t_offset[i])	
		else:
			Genome[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=genome)
			kernel[i].set_arg(0, Genome[i])
			Reads_fwd[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=reads_fwd[i])
			kernel[i].set_arg(1, Reads_fwd[i])
			Reads_rev_cmp[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=reads_rev[i])
			kernel[i].set_arg(2, Reads_rev_cmp[i])
			buffer_SA[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=SA)
			kernel[i].set_arg(3, buffer_SA[i])
			buffer_tally[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=tally)
			kernel[i].set_arg(4, buffer_tally[i])
			buffer_F[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=F)
			kernel[i].set_arg(5, buffer_F[i])
			cand_locs_per_read[i] = np.zeros(share_per_device[i], np.uint32)
			buffer_cand_locs_per_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cand_locs_per_read[i])
			kernel[i].set_arg(6, buffer_cand_locs_per_read[i])
			strand_of_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint8)
			buffer_strand_of_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=strand_of_read[i])
			kernel[i].set_arg(7, buffer_strand_of_read[i])
			mapped_endpos_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint32)
			buffer_mapped_endpos_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mapped_endpos_read[i])
			kernel[i].set_arg(8, buffer_mapped_endpos_read[i])
			buffer_t_offset[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=t_offset)
			kernel[i].set_arg(9, buffer_t_offset[i])		
		num_reads_alloted += share_per_device[i]		

	event = dict()
	start_time = time.time()
	for i in device_choices:
		event[i] = cl.enqueue_nd_range_kernel(chosen_device_parameters[i][1], kernel[i], (share_per_device[i],) , (num_work_items_per_device[i],))
		# event[i] = cl.enqueue_task(chosen_device_parameters[i][1], kernel[i], wait_for=None)

	for i in device_choices:
		event[i].wait()
		if(chosen_device_parameters[i][4] != "CPU"):
			print('Transfer operation involved')
			q1 = cl.enqueue_copy(chosen_device_parameters[i][1], cand_locs_per_read[i], buffer_cand_locs_per_read[i])
			q2 = cl.enqueue_copy(chosen_device_parameters[i][1], strand_of_read[i], buffer_strand_of_read[i])
			q3 = cl.enqueue_copy(chosen_device_parameters[i][1], mapped_endpos_read[i], buffer_mapped_endpos_read[i])
			q1.wait(); q2.wait(); q3.wait()
	total_time = time.time() - start_time
	# print('{:<25}{:<2}{:.4f}'.format('Mapping time',':',total_time),'s')
	# print(sum(cand_locs_per_read[0]), max(cand_locs_per_read[0]), min(cand_locs_per_read[0]), 'No of zeros =', len(cand_locs_per_read[0]) - np.count_nonzero(cand_locs_per_read[0])) #cand_locs_per_read[0],
	# print(sum(cand_locs_per_read[1]), max(cand_locs_per_read[1]), min(cand_locs_per_read[1]), 'No of zeros =', len(cand_locs_per_read[1]) - np.count_nonzero(cand_locs_per_read[1])) #cand_locs_per_read[0],
	print('{:<25}{:<2}{:.4f}'.format('Mapping time',':',total_time),'s')
	return strand_of_read, mapped_endpos_read, cand_locs_per_read


def find_defected_read(seq,e,n):
	count_A = 0; count_C = 0; count_G = 0; count_T = 0;count_N = 0; count_Random = 0
	fractional_read_len = (0.75*n)
	for i in seq:
		if(i == 'A'):
			count_A = count_A + 1
		elif(i == 'C'):
			count_C = count_C + 1
		elif(i == 'G'):
			count_G = count_G + 1
		elif(i == 'T'):
			count_T = count_T + 1
		elif(i == 'N'):
			count_N = count_N + 1
		else:
			count_Random = count_Random + 1

	if(count_Random > 0 or count_N > e or count_A > fractional_read_len or count_C > fractional_read_len or count_G > fractional_read_len or count_T > fractional_read_len):
		return 0
	else:
		return 1

def readFastq(filename, e, n):	
	sequence_names = []
	sequences = []
	qualities = []
	count = 0
	with open(filename) as fh:
		while True:
			name = fh.readline().rstrip()
			seq = fh.readline().rstrip()  # read base sequence
			fh.readline()  # skip placeholder line
			qual = fh.readline().rstrip() # base quality line
			if len(seq) == 0:
				break
			if(find_defected_read(seq,e,n) == 1):
				sequence_names.append(name)
				sequences.append(seq)
				qualities.append(qual)
	# print('Here1')
	return sequences, qualities, sequence_names

def main():	
	print('-----------------------------------------------------------------------')
	np.set_printoptions(threshold=sys.maxsize)
	# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
	p = subprocess.run(['pwd'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/'

	parser = argparse.ArgumentParser(description="PLEDGER: Pyopencl based tooL for gEnomic workloaDs tarGeting Embedded platfoRm")
	parser.add_argument("Fastq_filename", help="Give the name of reads file in fastq format.", type=str)
	parser.add_argument("Read_length", help="Provide the length of the reads. Choices: 100 or 150", type=int, default=100)
	parser.add_argument("Error", help="Provide the maximum permissible error. Smaller the value faster will be the algorithm. Range:[0-8]", type=int, default=5)
	parser.add_argument("Outputs", help="Maximum number of mappings allowed per read. Range:[1:3500]", type=int, default=100)
	parser.add_argument("klen", help="K-mer length. Range:[12:25]", type=int, default=12)
	parser.add_argument("-dc",help="Provide the numbers of all the devices to be used. These numbers can be from running 'get_device_choice.py' script", type=int, nargs='+')
	parser.add_argument("-rs",help="Provide the ratio for share of reads to be mapped for each device e.g. [0.6 0.2 0.2] for three devices. Sum of shares MUST be ONE.", type=float, nargs='+')
	parser.add_argument("-nr",help="Number of reads to be mapped, default: all the reads in the fastq file", type=int)
	parser.add_argument("-chr",help="Provide the serial numbers of chromosomes you want to map e.g. 1 2 21 would mean you want to map reads to chromosomes chr1, chr2 and chr21. To map to all the chromosomes, that were preprocessed, mention 'all'. The chromosomes include 1-22, X, Y", type=str, nargs='+')
	
	args = parser.parse_args()
	

	#-------Obtaining a list of chromosomes----------
	chr_list = []
	if(args.chr):
		chr_list = args.chr; #print(device_choices)
	else:
		print('ERROR: The chromosomes that need to be mapped not provided')
		sys.exit()

	for i in chr_list:
		if(i != '1' and i != '2' and i != '3' and i != '4' and i != '5' and i != '6' and i != '7' and i != '8' and i != '9' and i != '10' and i != '11' and i != '12' and i != '13' and i != '14' and i != '15' and i != '16' and i != '17' and i != '18' and i != '19' and i != '20' and i != '21' and i != '22' and i != 'X' and i != 'Y' and i != 'M' and i != 'all'):
			print('ERROR: Invalid chromosome number/name provided')
			sys.exit()
		if(len(chr_list) > 1 and (i == 'all')):
			print("ERROR: 'all' option provided along with individual chromosome name(s). Both are exclusive, either provide 'all' or list of chromosome numbers/names but not both.")
			sys.exit()
	
	chromosome_list = []		
	if(len(chr_list) == 1 and (chr_list[0] == 'all')):		
				
		chromosome_list = os.listdir(p+"F_FMIndex");	
		num_of_chromosomes = len(chromosome_list)
		chromosome_list = os.listdir(p+"SA");	
		if(len(chromosome_list) != num_of_chromosomes):
			print("ERROR: Some file(s) missing from folders storing preprocessed data. Please check the following folders: 'RefGenUppercase', 'F_FMIndex', 'SA', 'Tally_LMF, 'Tally_Offset'")
			sys.exit()
		chromosome_list = os.listdir(p+"Tally_LMF");	
		if(len(chromosome_list) != num_of_chromosomes):
			print("ERROR: Some file(s) missing from folders storing preprocessed data. Please check the following folders: 'RefGenUppercase', 'F_FMIndex', 'SA', 'Tally_LMF, 'Tally_Offset'")
			sys.exit()

		chromosome_list = os.listdir(p+"Tally_Offset");	
		if(len(chromosome_list) != num_of_chromosomes):
			print("ERROR: Some file(s) missing from folders storing preprocessed data. Please check the following folders: 'RefGenUppercase', 'F_FMIndex', 'SA', 'Tally_LMF, 'Tally_Offset'")
			sys.exit()
		chromosome_list = os.listdir(p+"RefGenUppercase");
		if(len(chromosome_list) != num_of_chromosomes):
			print("ERROR: Some file(s) missing from folders storing preprocessed data. Please check the following folders: 'RefGenUppercase', 'F_FMIndex', 'SA', 'Tally_LMF, 'Tally_Offset'")
			sys.exit()

		chromosome_list = [i.replace('_genome_uppercase.npy','') for i in chromosome_list]
		print('List of names of chromosomes to be mapped:')
		print(chromosome_list)
		print('Number of chromosomes :', num_of_chromosomes)

	else:
		for i in chr_list:
			chromosome_list.append('chr'+i)


	#------ Creating output folder------------
	dirName = 'PLEDGER_Output'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists")

	#---Read length
	n = args.Read_length
	if(n != 100 and n != 150):
		print('ERROR: Invalid read lengths, please check help.')
		sys.exit()	
	#---maximum permissible error
	e = args.Error
	if(e < 0 or e > 8):
		print('ERROR: Permissible error out of range:[0-7], please check help.')
		sys.exit()

	#-- q-gram length------
	q_len = args.klen
	max_klen = math.floor(n/(e+1))
	#print(max_klen)
	if(q_len > max_klen or q_len < 12):
		print('ERROR: K-mer length not acceptable for pigeonhole principle. Or it should be within range:[12-25], Default: 12, please check help.')
		sys.exit()
	#---Maximum number of mappings allowed per read, determines the memory usage on the device. If not enough memory available, reduce this number.
	no_of_outputs_per_read = args.Outputs
	if(no_of_outputs_per_read > 3500 or no_of_outputs_per_read < 1):
		print('ERROR: Number of outputs desired per read out of range: [1:3500], please check help')
		sys.exit()
	#---Device number 
	if(args.dc):
		device_choices = args.dc; #print(device_choices)
	else:
		print('ERROR: Device choice/s not provided.')
		sys.exit()
	#---Print OpenCL version
	print('{:<30}'.format('PyOpenCL version'),':',cl.VERSION_TEXT)	
	#---Obtain reads from fastq file
	sequences, qualities, sequence_names = readFastq(p + args.Fastq_filename, e, n) #, sequence_names, sequences_fwd, sequences_rev
	#---Number of reads to map
	if(args.nr and args.nr < len(sequences)):
		no_of_reads = args.nr
	else:
		no_of_reads = len(sequences)
	sequences = sequences[0:no_of_reads]

	#---Share per device
	if(args.rs):
		share_ratio_per_device = args.rs
	else:
		share_ratio_per_device = [1]

	#-------------------------------------------------------------------
	opencl(n, e, device_choices, no_of_reads, no_of_outputs_per_read, share_ratio_per_device, q_len, chromosome_list, p, sequence_names, args.Fastq_filename.split('.')[0], sequences) 
	print('-----------------------------------------------------------------------')


if __name__ == '__main__':
	main()