
import argparse
import sys
import re

def parse_args():
	parser = argparse.ArgumentParser(description="Higashi CpG_density")
	parser.add_argument('-g', '--genome_reference', type=str, default="./hg38.fa")
	parser.add_argument('-w', '--window', type=int, default='1000000')
	parser.add_argument('-o', '--output', type=str, default="./cpg_density.txt")
	return parser.parse_args()


def cal_cpg(bin_str):
	cg_count = bin_str.count("CG")
	# Each N leads to 2 2-mer with "N": "-N" and "N-"
	N_count = bin_str.count("N")
	# However if there are 2 "N" next to each other, you would overcount the number of 2-mers that have "N"
	
	# NN_count = 0
	# for i in range(len(bin_str) - 1):
	# 	if bin_str[i:i+2] == 'NN':
	# 		NN_count += 1
	NN_count = len(re.findall(r'(N)(?=\1)', bin_str))
	
	total_count = len(bin_str) - 1
	total_count = total_count - N_count * 2 + NN_count
	if bin_str[0] == 'N':
		total_count += 1
	elif bin_str[-1] == 'N':
		total_count += 1
	
	if total_count > 0:
		rate =  cg_count / (total_count)
		if rate < 0:
			print(cg_count, len(bin_str), N_count, NN_count)
			print(str)
			raise EOFError
	else:
		rate = 0.0
	
	
	return rate
args = parse_args()

f = open(args.genome_reference, "r")
line = f.readline()
output = open(args.output, "w")

bin_count = 0
chrom = ""
write_count = 0
bin_str = ""
chrom_length = {}
length = 0
while line:
	line = line.strip()
	
	if (line[0] == '>'):
		if len(chrom) > 0:
			output.write("%s\t%d\t%f\n" % (chrom, bin_count * args.window, cal_cpg(bin_str)))
			# chrom_length[chrom] = length
			# length = 0
		chrom = line[1:]
		bin_count = 0
		bin_str = ""
	else:
		line = line.upper()
		length += len(line)
		if len(bin_str) + len(line) == args.window:
			bin_str += line
			output.write("%s\t%d\t%f\n" % (chrom, bin_count * args.window, cal_cpg(bin_str)))
			bin_count += 1
			bin_str = ""
		elif len(bin_str) + len(line) > args.window:
			line1 = line[:args.window - len(bin_str)]
			line2 = line[args.window - len(bin_str):]
			bin_str += line1
			output.write("%s\t%d\t%f\n" % (chrom, bin_count * args.window, cal_cpg(bin_str)))
			bin_count += 1
			bin_str = line2
			
		else:
			bin_str += line
			
	if write_count % 10000 == 0:
		sys.stdout.flush()
		print("Process %d reads\r" % (write_count), end="")
	write_count += 1
	line = f.readline()
output.close()
f.close()
# print (chrom_length)
		