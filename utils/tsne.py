#
# script to use tsne to visualize results
#

from sklearn.manifold import TSNE 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from utils import load_data
from dataset import DataContainer as DC

# DATASET: datacontainer type holding RNAseq data
# LABELS_TO_USE: list of integer indices for classes to include
def tsne_viz(dataset, labels_to_use):
	tsne_data = []
	labels = []
	len_data = []

	for l in labels_to_use:
		indices = np.where(np.argmax(dataset.train.labels,axis=1) == l)[0]
		len_data.append(indices.shape[0])
		tsne_data.append(dataset.train.data[indices])
		labels.append(str(dataset.label_names_ordered[l]))
	data_t = np.vstack(tsne_data)

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

		print('start: ' + str(start))
		print('end: ' + str(end))
		
		data_separate.append(t_data[start:end])
		start += i

	print('plotting')
	fig, ax = plt.subplots()

	#colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
	colors = cm.rainbow(np.linspace(0, 1, len(labels_to_use)))

	for d, c, l in zip(data_separate, colors, labels):
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

	dataset = DC(data, total_gene_list, train_split=100, test_split=0)

	tsne_viz(dataset, [10, 12, 14, 16, 18, 20])#, 25, 40, 43, 50, 52])
