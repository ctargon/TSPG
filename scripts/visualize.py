import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn.manifold
import sklearn.preprocessing

import utils



def plot_tsne(x, y, classes, class_indices, x_perturbed=None, y_perturbed=-1):
	# extract data for each class into separate arrays
	tsne_n = []
	tsne_x = []
	tsne_y = []

	for class_index in class_indices:
		indices = (y == class_index)
		tsne_n.append(len(x[indices]))
		tsne_x.append(x[indices])
		tsne_y.append(classes[class_index])

	# append perturbed data if it was provided
	if x_perturbed is not None:
		n_perturbed = 100
		indices = np.arange(len(x_perturbed))
		np.random.shuffle(indices)
		class_indices.append(y_perturbed)
		tsne_n.append(n_perturbed)
		tsne_x.append(x_perturbed[indices[0:n_perturbed]])
		tsne_y.append("%s (perturbed)" % (classes[y_perturbed]))

	# perform t-SNE on merged data
	x_tsne = np.vstack(tsne_x)
	x_tsne = sklearn.manifold.TSNE().fit_transform(x_tsne)

	# separate embedded data back into separate arrays
	tsne_x = []
	start = 0
	for n in tsne_n:
		tsne_x.append(x_tsne[start:(start + n)])
		start += n

	# plot t-SNE embedding by class
	fig, ax = plt.subplots()
	colors = cm.rainbow(np.linspace(0, 1, len(class_indices)))

	for x, y, c in zip(tsne_x, tsne_y, colors):
		if "(perturbed)" in y:
			c = "k"
			alpha = 0.25
		else:
			alpha = 0.75

		ax.scatter(x[:, 0], x[:, 1], label=y, color=c, alpha=alpha)

	ax.legend(prop={"size": 6})
	ax.set_axisbelow(True)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.get_xaxis().set_ticklabels([])
	ax.get_yaxis().set_ticklabels([])
	ax.xaxis.set_ticks_position("none")
	ax.yaxis.set_ticks_position("none")
	plt.subplots_adjust(right=0.7)
	plt.grid(b=True, which="major", alpha=0.3)
	plt.show()



if __name__ == "__main__":
	# parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="input dataset (samples x genes)", required=True)
	parser.add_argument("--labels", help="list of sample labels", required=True)
	parser.add_argument("--gene-sets", help="list of curated gene sets")
	parser.add_argument("--target", help="target class of perturbed data", type=int, default=-1)

	args = parser.parse_args()

	# load input data
	print("loading input dataset...")

	df = utils.load_dataframe(args.dataset)
	df_samples = df.index
	df_genes = df.columns

	labels, classes = utils.load_labels(args.labels)

	print("loaded input dataset (%s genes, %s samples)" % (df.shape[1], df.shape[0]))

	# print target class if specified
	if args.target != -1:
		print("target class is: %s" % (classes[args.target]))

	# load gene sets file if it was provided
	if args.gene_sets != None:
		print("loading gene sets...")

		gene_sets = utils.load_gene_sets(args.gene_sets)

		print("loaded %d gene sets" % (len(gene_sets)))

		# remove genes which do not exist in the dataset
		genes = list(set(sum([genes for (name, genes) in gene_sets], [])))
		missing_genes = [g for g in genes if g not in df_genes]

		gene_sets = [(name, [g for g in genes if g in df_genes]) for (name, genes) in gene_sets]

		print("%d / %d genes from gene sets were not found in the input dataset" % (len(missing_genes), len(genes)))
	else:
		gene_sets = []

	# create t-SNE visualization for each gene set
	for name, genes in gene_sets:
		# extract dataset
		x = df[genes]
		y = labels

		# normalize dataset
		x = sklearn.preprocessing.MinMaxScaler().fit_transform(x)

		# select classes to include in plot
		class_indices = list(range(len(classes)))

		# plot t-SNE visualization
		if args.target != -1:
			x_perturbed = np.load("perturbed_%d.npy" % (args.target))

			plot_tsne(x, y, classes, class_indices, x_perturbed, args.target)
		else:
			plot_tsne(x, y, classes, class_indices)
