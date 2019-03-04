import json
import re
import itertools
import os, sys, argparse
import numpy as np

from dataset import DataContainer as DC


def parse_and_load_data():
	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run classification on specified dataset, \
		subset of genes, or a random set')
	parser.add_argument('--dataset', help='dataset to be used', type=str, required=True)
	parser.add_argument('--gene_list', help='list of genes in dataset (same order as dataset)', \
		type=str, required=True)
	parser.add_argument('--class_counts', help='json file containing number of samples per class', \
		type=str, required=True)
	parser.add_argument('--subset_list', help='gmt/gct file containing subsets', type=str, required=False)
	parser.add_argument('--set', help='specific subset to run', type=str, required=False)
	parser.add_argument('--target', help='target class', type=int, required=False, default=-1)

	args = parser.parse_args()

	# load the data
	print('loading genetic data...')
	gtex_gct_flt = np.load(args.dataset)
	total_gene_list = np.load(args.gene_list)
	data = load_data(args.class_counts, gtex_gct_flt)

	# if subset is passed, filter out the genes that are not in the total gene list
	# and redefine the subsets with valid genes
	if args.subset_list:
		subsets = read_subset_file(args.subset_list)

		tot_genes = []
		missing_genes = []

		print('checking for valid genes...')
		for s in subsets:
			genes = []
			for g in subsets[s]:
				if g not in tot_genes:
					tot_genes.append(g)
				if g in total_gene_list:
					genes.append(g)
				else:
					if g not in missing_genes:
						missing_genes.append(g)
			subsets[s] = genes
					#print('missing gene ' + str(g))
		print('missing ' + str(len(missing_genes)) + '/' + str(len(tot_genes)) + ' genes' + ' or ' \
			 + str(int((float(len(missing_genes)) / len(tot_genes)) * 100.0)) + '% of genes')


	if args.subset_list:
		# dataset using only certain genes
		dataset = DC(data, total_gene_list, subsets[args.set.upper()])
	else:
		# dataset using every gene
		dataset = DC(data, total_gene_list)

	return dataset, args.target


# USAGE:
#   -helper function to print out the classes/number corresponding to the class
def print_class_counts(class_counts_file):
	cc_dict = json.load(open(class_counts_file))
	for i, k in zip(enumerate(sorted(cc_dict)), (sorted(cc_dict))):
		print("{}: {} {}".format(i[0], str(k), f[k]))


# get random gene indexes between 0-len total_gene_list
def create_random_subset(num_genes, total_gene_list):
	#Generate Gene Indexes for Random Sample
	gene_indexes = np.random.randint(0, len(total_gene_list), num_genes)
	return [total_gene_list[i] for i in gene_indexes]


def load_data(num_samples_json, gtex_gct_flt):
	sample_count_dict = {}
	with open(num_samples_json) as f:
		sample_count_dict = json.load(f)

	idx = 0
	data = {}

	for k in sorted(sample_count_dict.keys()):
		data[k] = gtex_gct_flt[:,idx:(idx + int(sample_count_dict[k]))]
		idx = idx + int(sample_count_dict[k])

	return data


# USAGE:
# 	- read a csv or txt file that contains a name of a subset followed by a list of genes
# PARAMS:
#	file: file to read
def read_subset_file(file):
	with open(file, 'r') as f:
		content = f.readlines()

	# eliminate new line characters
	content = [x.strip() for x in content]

	# split on tabs or commas to create a sublist of set names and genes
	content = [re.split('\t|,| ', x) for x in content]

	# create a dictionary with keys subset names and values list of genes
	subsets = {}
	for c in content:
		subsets[c[0]] = c[1:]

	return subsets


# write a subset list dictionary to a file
def write_subsets_to_file(subsets, file):
	with open(file, 'w') as f:
		for s in subsets:
			f.write(s + '\t')
			for g in subsets[s]:
				f.write(g + '\t')
			f.write('\n')

