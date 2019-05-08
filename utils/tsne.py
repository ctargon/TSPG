#
# script to use tsne to visualize results
#

from sklearn.manifold import TSNE 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn import preprocessing
import argparse

import umap

from utils import load_data, read_subset_file
from dataset import DataContainer as DC

# DATASET: datacontainer type holding RNAseq data
# LABELS_TO_USE: list of integer indices for classes to include
def tsne_viz(dataset, labels_to_use, perturbed=None, perturbed_label=-1):
	tsne_data = []
	labels = []
	len_data = []

	np.set_printoptions(precision=4, suppress=True)

	for l in labels_to_use:
		indices = np.where(np.argmax(dataset.train.labels,axis=1) == l)[0]
		len_data.append(indices.shape[0])
		tsne_data.append(dataset.train.data[indices])
		labels.append(str(dataset.label_names_ordered[l]))


	if (perturbed is not None):
		num_perturbed_samples = 100
		r_idxs = np.arange(perturbed.shape[0])
		np.random.shuffle(r_idxs)
		tsne_data.append(perturbed[r_idxs[:num_perturbed_samples]])
		labels_to_use.append(perturbed_label)
		len_data.append(num_perturbed_samples)
		labels.append('Perturbed ' + str(dataset.label_names_ordered[perturbed_label]))
	
	data_t = np.vstack(tsne_data)
	print(data_t.shape)

	print('calculating tsne...')
	tsne = TSNE()
	t_data = tsne.fit_transform(data_t)
	# t_data = umap.UMAP().fit_transform(data_t)

	data_separate = []
	start = 0
	total = 0
	print(t_data.shape)
	print(len_data)
	for i in len_data:
		end = start + i
		data_separate.append(t_data[start:end])
		start += i

	print('plotting')
	fig, ax = plt.subplots()

	colors = cm.rainbow(np.linspace(0, 1, len(labels_to_use)))

	for d, c, l in zip(data_separate, colors, labels):
		if "PERTURB" in l.upper():
			c = 'k'
			ax.scatter(d[:,0], d[:,1], color=c, label=l, alpha=0.25)
			continue
		ax.scatter(d[:,0], d[:,1], color=c, label=l, alpha=0.75)


	ax.legend(prop={'size': 6})
	ax.set_axisbelow(True)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.get_xaxis().set_ticklabels([])
	ax.get_yaxis().set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	plt.subplots_adjust(right=0.7)
	plt.grid(b=True, which='major', alpha=0.3)

	plt.show()






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run classification on specified dataset, \
		subset of genes, or a random set')
	parser.add_argument('--dataset', help='GEM to be used', type=str, required=True)
	parser.add_argument('--gene_list', help='list of genes in dataset (same order as dataset)', \
		type=str, required=True)
	parser.add_argument('--class_counts', help='json file containing number of samples per class', \
		type=str, required=True)
	parser.add_argument('--subset_list', help='gmt/gct file containing subsets', type=str, required=False)
	parser.add_argument('--set', help='specific subset to run', type=str, required=False)
	parser.add_argument('--perturbed', help='index of perturbed data to load \
											from ./data/perturbed', type=int, required=False)

	args = parser.parse_args()

	# load data
	print('loading data...')
	gem = np.load(args.dataset)
	total_gene_list = np.load(args.gene_list)
	data = load_data(args.class_counts, gem)

	subset = args.set

	if subset:
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

	if subset:
		# dataset using only certain genes
		dataset = DC(data, total_gene_list, subsets[subset.upper()], train_split=100, test_split=0)
	else:
		# dataset using every gene
		dataset = DC(data, total_gene_list, train_split=100, test_split=0)

	# preprocess data
	scaler = preprocessing.MinMaxScaler() #preprocessing.MaxAbsScaler()
	dataset.train.data = scaler.fit_transform(dataset.train.data)
	#dataset.test.data = scaler.fit_transform(dataset.test.data)

	rand_idxs = np.arange(0, dataset.num_classes)
	np.random.shuffle(rand_idxs)
	rand_idxs = list(rand_idxs[:10])

	if args.perturbed:
		pert_idx = args.perturbed
		if pert_idx not in rand_idxs:
			rand_idxs.pop()
			rand_idxs.append(pert_idx)

		perturbed = np.load("./data/perturbed/perturbed_" + str(pert_idx) + ".npy")

		tsne_viz(dataset, rand_idxs, perturbed, pert_idx)
	else:
		tsne_viz(dataset, arange(dataset.num_classes))
