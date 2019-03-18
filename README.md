# Transcriptome State Perturbation Generator
WIP...

## Notes
Using the DataContainer module means you must follow some strict guidelines:
* GEM is organized in ALPHABETICAL order by class name (rows are genes, columns are samples)...
	* e.g. columns would be | ARTERY ARTERY ARTERY HEART HEART HEART PANCREAS PANCREAS | for a 3 class GEM
* You must provide a json "counts" file, containing the class names with the number of counts
	* DataContainer creates labels for you automatically using this "counts" file
If you follow these rules, the DataContainer module is your friend. Otherwise, your model will be learning incorrect labels.