#!/usr/bin/env python3

import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.manifold
import sklearn.preprocessing

import utils



def cleanse_label(label):
	label = label.replace(" ", "_")
	label = label.replace("-", "")
	label = label.replace("(", "")
	label = label.replace(")", "")
	return label



def plot_tsne(x, y, classes, class_indices, x_perturbed=None, y_perturbed=-1, output_dir="."):
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

	plt.savefig("%s/tsne.png" % (output_dir))
	plt.close()



def plot_heatmap(df, source_class, target_class, output_dir="."):
	fig, ax = plt.subplots(1, len(df.columns))

	# create user-defined colormap
	cdict = {
		"red": (
			(0.0, 0.0, 0.0),
			(0.25, 0.0, 0.0),
			(0.5, 0.8, 1.0),
			(0.75, 1.0, 1.0),
			(1.0, 0.4, 1.0)
		),
		"green": (
			(0.0, 0.0, 0.0),
			(0.25, 0.0, 0.0),
			(0.5, 0.9, 0.9),
			(0.75, 0.0, 0.0),
			(1.0, 0.0, 0.0)
		),
		"blue": (
			(0.0, 0.0, 0.4),
			(0.25, 1.0, 1.0),
			(0.5, 1.0, 0.8),
			(0.75, 0.0, 0.0),
			(1.0, 0.0, 0.0)
		)
	}
	blue_red = mcol.LinearSegmentedColormap("BlueRed1", cdict)
	plt.register_cmap(name="BlueRed", cmap=blue_red)

	for i in range(len(df.columns)):
		# get the vector, then tile it some so it is visible if very long
		column = np.expand_dims(df[df.columns[i]], -1)
		column = np.tile(column, (1, max(1, int(column.shape[0] / 10))))

		# plot tiled vector as heatmap image
		im = ax[i].imshow(column, cmap="BlueRed")
		im.set_clim(-1, 1)

		ax[i].set_title(df.columns[i])
		ax[i].set_xticks([])
		ax[i].set_xticklabels([])

		# display row names if there aren't too many
		if i == 0 and len(df.index) < 30:
			ax[i].set_yticks(np.arange(len(df.index)))
			ax[i].set_yticklabels(df.index)
		else:
			ax[i].set_yticks([])
			ax[i].set_yticklabels([])

	# insert colobar and shrink it some
	cbar = ax[-1].figure.colorbar(im, ax=ax[-1], shrink=0.5)
	cbar.ax.set_ylabel("Expression Level", rotation=-90, va="bottom")

	plt.savefig("%s/%s_to_%s.png" % (output_dir, source_class, target_class))
	plt.close()



if __name__ == "__main__":
	# parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-data", help="training data (samples x genes)", required=True)
	parser.add_argument("--train-labels", help="training labels", required=True)
	parser.add_argument("--test-data", help="test data (samples x genes)", required=True)
	parser.add_argument("--test-labels", help="test labels", required=True)
	parser.add_argument("--gene-sets", help="list of curated gene sets")
	parser.add_argument("--set", help="specific gene set to run")
	parser.add_argument("--tsne", help="plot t-SNE of samples", action="store_true")
	parser.add_argument("--heatmap", help="plot heatmaps of sample perturbations", action="store_true")
	parser.add_argument("--target", help="target class of perturbed data", type=int, default=-1)
	parser.add_argument("--output-dir", help="Output directory", default=".")

	args = parser.parse_args()

	# load input data
	print("loading train/test data...")

	df_train = utils.load_dataframe(args.train_data)
	df_test = utils.load_dataframe(args.test_data)

	y_train, classes = utils.load_labels(args.train_labels)
	y_test, _ = utils.load_labels(args.test_labels, classes)

	print("loaded train data (%s genes, %s samples)" % (df_train.shape[1], df_train.shape[0]))
	print("loaded test data (%s genes, %s samples)" % (df_test.shape[1], df_test.shape[0]))

	# impute missing values
	min_value = df_train.min().min()

	df_train.fillna(value=min_value, inplace=True)
	df_test.fillna(value=min_value, inplace=True)

	# print target class if specified
	if args.target != -1:
		print("target class is: %s" % (classes[args.target]))

	# load gene sets file if it was provided
	if args.gene_sets != None:
		print("loading gene sets...")

		gene_sets = utils.load_gene_sets(args.gene_sets)
		gene_sets = utils.filter_gene_sets(gene_sets, df_test.columns)

		print("loaded %d gene sets" % (len(gene_sets)))
	else:
		gene_sets = {"all_genes": df_genes}

	# create visualizations for each gene set
	name = args.set

	try:
		genes = gene_sets[name]
	except:
		print("gene set is not the subset file provided")
		sys.exit(1)

	# extract train/test data
	x_train = df_train[genes]
	x_test = df_test[genes]

	# normalize test data (using the train data)
	scaler = sklearn.preprocessing.MinMaxScaler()
	scaler.fit(x_train)

	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	# select classes to include in plot
	class_indices = list(range(len(classes)))

	# plot t-SNE visualization if specified
	if args.tsne:
		if args.target != -1:
			x_perturbed = np.load("%s/perturbed_%d.npy" % (args.output_dir, args.target))

			plot_tsne(x_train, y_train, classes, class_indices, x_perturbed, args.target, output_dir=args.output_dir)
		else:
			plot_tsne(x_train, y_train, classes, class_indices, output_dir=args.output_dir)

	# plot heatmaps for each source-target pair if specified
	if args.heatmap:
		for i in range(len(classes)):
			try:
				# load pertubation data
				source_class = cleanse_label(classes[i])
				target_class = cleanse_label(classes[args.target])

				data = np.load("%s/%s_to_%s.npy" % (args.output_dir, source_class, target_class))
				data = data.T

				# initialize dataframe
				df_pert = pd.DataFrame(data, index=genes, columns=["X", "P", "X_adv", "mu_T"])

				# sort genes by perturbation value
				df_pert = df_pert.sort_values("P", ascending=False)

				# plot heatmap of perturbation data
				plot_heatmap(df_pert, source_class, target_class, output_dir=args.output_dir)
			except FileNotFoundError:
				print("warning: no data found for %s to %s" % (source_class, target_class))
