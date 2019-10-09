import argparse
import sklearn.model_selection
import sklearn.preprocessing

import utils
from target_models import Target_A as Target



if __name__ == "__main__":
	# parse command-line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="input dataset (samples x genes)", required=True)
	parser.add_argument("--labels", help="list of sample labels", required=True)
	parser.add_argument("--gene-sets", help="list of curated gene sets")
	parser.add_argument("--set", help="specific gene set to run")
	parser.add_argument("--output-dir", help="Output directory", default=".")

	args = parser.parse_args()

	# load input data
	print("loading input dataset...")

	df = utils.load_dataframe(args.dataset)
	df_samples = df.index
	df_genes = df.columns

	labels, classes = utils.load_labels(args.labels)

	print("loaded input dataset (%s genes, %s samples)" % (df.shape[1], df.shape[0]))

	# impute missing values
	df.fillna(value=df.min().min(), inplace=True)

	# load gene sets file if it was provided
	if args.gene_sets != None:
		print("loading gene sets...")

		gene_sets = utils.load_gene_sets(args.gene_sets)

		print("loaded %d gene sets" % (len(gene_sets)))

		# remove genes which do not exist in the dataset
		genes = list(set(sum([gene_sets[name] for name in gene_sets.keys()], [])))
		missing_genes = [g for g in genes if g not in df_genes]

		gene_sets = {name: [g for g in genes if g in df_genes] for name, genes in gene_sets.items()}

		print("%d / %d genes from gene sets were not found in the input dataset" % (len(missing_genes), len(genes)))
	else:
		gene_sets = {"all_genes": df_genes}

	# train a model for each gene set
	name = args.set
	genes = gene_sets[name]
	# initialize output directory
	output_dir = "%s/%s" % (args.output_dir, name)

	# extract dataset
	X = df[genes]
	y = utils.onehot_encode(labels, classes)

	# create train/test sets
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

	# normalize dataset
	Scaler = sklearn.preprocessing.MinMaxScaler
	x_train = Scaler().fit_transform(x_train)
	x_test = Scaler().fit_transform(x_test)

	clf = Target(n_input=x_train.shape[1], n_classes=len(classes), epochs=30, batch_size=32, output_dir=output_dir)
	clf.train(x_train, y_train)
	clf.inference(x_test, y_test)
