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
		im = ax[i].imshow(np.expand_dims(arr[i], -1), cmap="BlueRed")
		im.set_clim(-1,1)

		ax[i].set_title(titles[i])

		ax[i].set_xticks([])
		ax[i].set_xticklabels([])

		if i == 0:
			ax[i].set_yticks(np.arange(arr.shape[1]))
			ax[i].set_yticklabels(genes)
		else:
			ax[i].set_yticks([])
			ax[i].set_yticklabels([])

	cbar = ax[i].figure.colorbar(im, ax=ax[i], shrink=0.5)
	cbar.ax.set_ylabel("Expression Level", rotation=-90, va="bottom")

	plt.show()



if __name__ == "__main__":
	results = np.load("./data/heatmap/Adipose__Visceral_Omentum_to_Ovary.npy")

	# get list of genes
	total_gene_list = np.load("./data/tissue/gtex_gene_list_v7.npy")
	subsets = read_subset_file("./data/subsets/hallmark_experiments.txt")

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

	genes = subsets["HALLMARK_HEDGEHOG_SIGNALING"]
	names = ["X", "P", "X_adv", "mu_T"]

	# create dataframe with gene names and 4 vectors
	df = pd.DataFrame(data=results.T, columns=names, index=genes)

	# sort dataframe by perturbation
	df = df.sort_values("P", ascending=False)

	heatmap(df.values.T, df.index.values.tolist())




