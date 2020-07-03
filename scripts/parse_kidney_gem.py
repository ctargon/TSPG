#
# script used to parse and separate the kidney gem
#

import pandas as pd 
import numpy as np 
import os, sys


if __name__ == "__main__":
	# read in kidney/annotation files
	kidney = pd.read_csv("./data/kidney/Kidney_FPKM.tab", sep='\t')
	ann = pd.read_csv("./data/kidney/Annotation_collapsed.txt", sep='\t')

	# get indices of healthy samples and tumor samples
	healthy_samples = ann.loc[ann["Tissue"] == "Solid Tissue Normal", "Sample"].values
	tumor_samples = ann.loc[ann["Tissue"] == "Primary Tumor", "Sample"].values

	# extract healthy gem
	healthy_gem = kidney[healthy_samples]

	# get annotation file just for tumors and get cancer types
	tumor_annotation = ann.loc[ann["Sample"].isin(tumor_samples)]
	cancer_types = np.unique(tumor_annotation["Cancer"].values)

	# first save the GEMs of each cancer type and collect annotations for each type
	cancer_anns = {}
	for c in cancer_types:
		samples = tumor_annotation.loc[tumor_annotation["Cancer"] == c, "Sample"].values
		cancer_anns[c] = tumor_annotation.loc[tumor_annotation["Sample"].isin(samples)]

		gem = kidney[samples]
		os.makedirs("./data/kidney/gems/" + c)
		np.save("./data/kidney/gems/" + c + "/" + c + "_all.npy", gem.values)

	# find cancer progression unique vals, iterate through for each cancer type
	cancer_progressions = np.unique(tumor_annotation["TumorStage"].values)

	for k in cancer_anns:
		for p in cancer_progressions:
			samples = cancer_anns[k].loc[cancer_anns[k]["TumorStage"] == p, "Sample"].values
			if samples.shape[0]:
				gem = df[samples]
				path = "./data/kidney/gems/" + k + "/" + k + "_" + p.replace(" ", "_") + ".npy"
				np.save(path, gem.values)

