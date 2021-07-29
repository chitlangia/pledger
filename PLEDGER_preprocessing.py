import numpy as np
import sys
import argparse
import subprocess
import time
import math
import os

def convert_char_to_int_in_genome(genome):
	gen = []
	template_dict = {'A':0,'C':1, 'G':2,'T':3, 'Z':5, '$':5}
	for i in genome:
		gen.append(template_dict[i])
	return gen


def readGenome(f):
	genome = ''
	for line in f:
		if not line[0] == '>':
			genome += line.rstrip()
	return genome


def RefGenomePreparation(f, chromosome_name):
	genome = readGenome(f)
	gen = ''
	for s in genome:
		if s == 'N' or s == 'n':
			s = 'Z'		
		gen += s.upper()	
	genome = convert_char_to_int_in_genome(gen)	
	np.save('./RefGenUppercase/'+ chromosome_name +'_genome_uppercase.npy',np.asarray(genome,np.uint8), allow_pickle=False)
	genome.append(5)
	genome = np.asarray(genome,np.uint8)
	gen = gen+'$'
	file = open(chromosome_name,'w')
	file.write(gen) 
	file.close()	
	return genome


def FMIndexViaSA(genome, SA):
	print('Constructing L array')
	L = np.zeros((len(SA),), dtype=np.uint8)#np.chararray((len(SA),))
	for si in range(len(SA)):
		if SA[si] == 0:
			L[si] = 5
		else:
			L[si] = genome[SA[si]-1]
	print('L array constructed')
	return L


def build_F(tally, chromosome_name):
	# F array stores the cummulative sum of alphabets in the sequence A, C, G and T
	F = np.zeros(5,np.uint32)   # 4 + 1, where 4 is the number of alphabets and remaining one is the cummulative sum of all the characters.
	F[0] = 1
	for i in range(1,len(F)):
		F[i] = F[i-1] + tally[i-1]	
	np.save('./F_FMIndex/'+chromosome_name+'_F_FMIndex.npy',F, allow_pickle=False)


def tally_offset_array(L, chromosome_name):
	print('Constructing Tally Offset')
	t_offset = np.zeros((len(L),), dtype=np.uint16)
	for i in range(0,len(L),16):
		a = 0; c = 0; g = 0; t = 0;
		if(i+16 < len(L)):
			up_limit = i+16			
		else:
			up_limit = len(L)
		for j in range(i+1,up_limit):
			offset = 0
			if(L[j] == 0):
				a = a + 1
			elif(L[j] == 1):
				c = c + 1
			elif(L[j] == 2):
				g = g + 1
			elif(L[j] == 3):
				t = t + 1
			offset = offset|a;
			offset = offset|(c << 4)
			offset = offset|(g << 8)
			offset = offset|(t << 12)
			t_offset[j] = offset
	np.save('./Tally_Offset/'+ chromosome_name +'_Tally_offset.npy',t_offset, allow_pickle=False)
	print('Tally Offset constructed')


def build_tally_matrix(L, chromosome_name):
	tally_offset_array(L, chromosome_name)
	tally_prev = np.zeros((4,),np.uint32)
	tally_current = np.zeros((4,),np.uint32)
	tally_16th = np.zeros((int(len(L)/16) + 1,4),np.uint32)
	print('Constructing Tally Datastructure')
	j = 0;
	if(L[0] != 5):
		tally_prev[L[0]] = 1
		tally_16th[j][0]  = tally_prev[0]
		tally_16th[j][1]  = tally_prev[1]
		tally_16th[j][2]  = tally_prev[2]
		tally_16th[j][3]  = tally_prev[3]
	j = j+1
	for i in range(1,len(L)):#
		tally_current[0] = tally_prev[0];tally_current[1] = tally_prev[1];tally_current[2] = tally_prev[2];tally_current[3] = tally_prev[3]	
		if(L[i] != 5):
			tally_current[L[i]] += 1
		tally_prev[0] = tally_current[0];tally_prev[1] = tally_current[1];tally_prev[2] = tally_current[2];tally_prev[3] = tally_current[3]	
		if((i%16)==0):
			tally_16th[j][0]  = tally_prev[0]
			tally_16th[j][1]  = tally_prev[1]
			tally_16th[j][2]  = tally_prev[2]
			tally_16th[j][3]  = tally_prev[3]
			j = j+1

	np.save('./Tally_LMF/'+ chromosome_name+'_Tally_LMF.npy',tally_16th, allow_pickle=False)  #LMF - Low Memory Footprint
	print('Tally data structure constructed')
	build_F(tally_current, chromosome_name)

def main():
	print('\n');print('---------------------------------------------------------------------------')
	print('This process is not parallelised, and hence, takes time. Please wait or use our pre uploaded data.')
	np.set_printoptions(threshold=sys.maxsize)

	p = subprocess.run(['pwd'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/'

	parser = argparse.ArgumentParser(description="Preprocessing module for PLEDGER: Pyopencl based tooL for gEnomic workloaDs tarGeting Embedded cluster")
	parser.add_argument("Ref_Gen_Name", help="Provide the folder and reference genome file name e.g. Genome/chr21", type=str)
	parser.add_argument("Output_File_Name", help="Provide the chromosome name e.g. chr2 or chr21.", type=str)
	args = parser.parse_args()	

	start_time = time.time()
	dirName = 'RefGenUppercase'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists")

	dirName = 'Tally_LMF'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists")


	dirName = 'Tally_Offset'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists")


	dirName = 'F_FMIndex'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists")


	dirName = 'SA'
	try:
		# Create target Directory
		os.mkdir(dirName)
		print("Directory " , dirName ,  " Created ") 
	except FileExistsError:
		print("Directory " , dirName ,  " already exists")

	with open(p + args.Ref_Gen_Name,'r') as f:
		genome = RefGenomePreparation(f, args.Output_File_Name)

	##------------------------------------------------------------------------------------------
	subprocess.call(['./suftest', args.Output_File_Name]);
	SA = np.zeros((len(genome),),dtype=np.uint32)
	j = 0;
	with open(args.Output_File_Name + '_SA.txt') as f:
		for line in f:
			temp = line.rstrip()
			temp = np.asarray(temp, dtype=np.uint32)
			SA[j] = temp
			j = j + 1
	
	np.save('./SA/'+args.Output_File_Name+'_SA.npy',SA, allow_pickle=False)
	subprocess.call(['rm',args.Output_File_Name + '_SA.txt']);
	# SA = SA[0]
	#------------------------------------------------------------------------------------------
	print('Suffix array built')
	L = FMIndexViaSA(genome, SA)
	build_tally_matrix(L, args.Output_File_Name)
	subprocess.call(['rm',args.Output_File_Name]);
	total_time = time.time() - start_time
	print('Length of reference genome =',len(genome)-1)
	print('Time for preprocessing =', total_time, 's')


if __name__ == '__main__':
	main()