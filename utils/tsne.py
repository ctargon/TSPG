#
# script to use tsne to visualize results
#

from sklearn.manifold import TSNE 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn import preprocessing

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
		labels.append('perturbed ' + str(dataset.label_names_ordered[perturbed_label]))
	
	data_t = np.vstack(tsne_data)
	print(data_t.shape)

	print('calculating tsne...')
	tsne = TSNE()
	t_data = tsne.fit_transform(data_t)

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

	#colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
	colors = cm.rainbow(np.linspace(0, 1, len(labels_to_use)))

	for d, c, l in zip(data_separate, colors, labels):
		if "PERTURB" in l.upper():
			c = 'k'
		ax.scatter(d[:,0], d[:,1], color=c, label=l)


	ax.legend(bbox_to_anchor=(1.01,1), prop={'size': 6})

	plt.subplots_adjust(right=0.7)

	plt.show()






if __name__ == '__main__':
	# load data
	print('loading data...')
	gtex_gct_flt = np.load("./data/gtex_gct_data_float_v7.npy")
	total_gene_list = np.load("./data/gtex_gene_list_v7.npy")
	data = load_data("./data/gtex_tissue_count_v7.json", gtex_gct_flt)

	perturbed = np.load("./data/perturbed_46.npy")

	subset = "HALLMARK_HEDGEHOG_SIGNALING"

	if subset:
		subsets = read_subset_file("./data/hallmark_experiments.txt")

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

	tsne_viz(dataset, [0, 2, 10, 19, 25, 33, 34, 36, 46, 48], perturbed, 46)
