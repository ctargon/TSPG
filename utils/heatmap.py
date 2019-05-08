import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.colors as mcol
import matplotlib.cm as cm

from utils import read_subset_file


def heatmap(arr, genes, titles=[r"$X$", r"$P$", r"$X_{adv}$", r"$\mu_{T}$"]):
	fig, ax = plt.subplots(1, 4)

	# Make a user-defined colormap.
	# cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
	cdict = {'red':  ((0.0, 0.0, 0.0),
				   (0.25, 0.0, 0.0),
				   (0.5, 0.8, 1.0),
				   (0.75, 1.0, 1.0),
				   (1.0, 0.4, 1.0)),

		 'green': ((0.0, 0.0, 0.0),
				   (0.25, 0.0, 0.0),
				   (0.5, 0.9, 0.9),
				   (0.75, 0.0, 0.0),
				   (1.0, 0.0, 0.0)),

		 'blue':  ((0.0, 0.0, 0.4),
				   (0.25, 1.0, 1.0),
				   (0.5, 1.0, 0.8),
				   (0.75, 0.0, 0.0),
				   (1.0, 0.0, 0.0))
	}
	blue_red = mcol.LinearSegmentedColormap('BlueRed1', cdict)
	plt.register_cmap(name="BlueRed", cmap=blue_red) 

	# cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)

	for i in range(4):
		# get the vector, then tile it some so it is visible if very long
		npimg = np.expand_dims(arr[i], -1)
		npimg = np.tile(npimg, (1,npimg.shape[0]/10))

		# plot tiled vector as heatmap image
		im = ax[i].imshow(npimg, cmap="BlueRed")
		im.set_clim(-1,1)

		ax[i].set_title(titles[i])

		ax[i].set_xticks([])
		ax[i].set_xticklabels([])

		# display gene names if the genes are less than 40 otherwise too crowded
		if i == 0 and len(genes) < 30:
			ax[i].set_yticks(np.arange(arr.shape[1]))
			ax[i].set_yticklabels(genes)
		else:
			ax[i].set_yticks([])
			ax[i].set_yticklabels([])

	# insert colobar and shrink it some
	cbar = ax[i].figure.colorbar(im, ax=ax[i], shrink=0.5)
	cbar.ax.set_ylabel("Expression Level", rotation=-90, va="bottom")

	plt.show()



if __name__ == "__main__":
	# get list of genes
	# total_gene_list = np.load("./data/kidney/kidney_gene_list.npy")
	total_gene_list = np.load("./data/tissue/gtex_gene_list_v7.npy")
	subsets = read_subset_file("./data/subsets/hallmark_experiments.txt")
	# subsets = read_subset_file("./data/subsets/random_experiments.txt")

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

	# genes = subsets["RANDOM50_2"]
	genes = subsets["HALLMARK_ALL"]
	#genes = subsets["HALLMARK_TNFA_SIGNALING_VIA_NFKB"]
	def printgenes(f, l):
		for i in range(len(l) - 1):
			f.write(l[i] + ', ')
		f.write(l[-1] + "\n")

	def print_top_genes_to_file(fw, indir):
		files = sorted(os.listdir(indir))
		files = [indir + "/" + f for f in files]

		results = []
		for f in files:
			results.append(np.load(f))

		myf = open(fw, "w")
		for r,f in zip(results,files):
			genes = subsets["HALLMARK_ALL"]
			names = ["X", "P", "X_adv", "mu_T"]

			# create dataframe with gene names and 4 vectors
			df = pd.DataFrame(data=r.T, columns=names, index=genes)

			# sort dataframe by perturbation
			df = df.sort_values("P", ascending=False)
			top = df.iloc[0:20].index.tolist()
			bottom =  df.iloc[-20:].index.tolist()

			myf.write(f + ", TOP, ")
			printgenes(myf, top)
			myf.write(f + ", BOTTOM, ")
			printgenes(myf, bottom)

	# print_top_genes_to_file("./KIRC-transition-genes.txt", "./data/heatmap/KIRC")

	results = np.load("./data/heatmap/Heart-Left-Ventricle/Thyroid_to_Heart__Left_Ventricle.npy")

	genes = subsets["HALLMARK_ALL"]
	names = ["X", "P", "X_adv", "mu_T"]

	# create dataframe with gene names and 4 vectors
	df = pd.DataFrame(data=results.T, columns=names, index=genes)

	# sort dataframe by perturbation
	df = df.sort_values("P", ascending=False)
	top = df.iloc[0:20].index.tolist()
	bottom =  df.iloc[-20:].index.tolist()
	heatmap(df.values.T, df.index.values.tolist())


