import copy
import numpy as np
import pandas as pd
import sys
import re



def split_filename(filename):
    tokens = filename.split('.')

    return '.'.join(tokens[:-1]), tokens[-1]



def load_dataframe(filename):
    basename, ext = split_filename(filename)

    if ext == 'txt':
        # load dataframe from plaintext file
        return pd.read_csv(filename, sep='\t', index_col=0)
    elif ext == 'npy':
        # load data matrix from binary file
        X = np.load(filename)

        # load row names and column names from text files
        rownames = np.loadtxt('%s.rownames.txt' % basename, dtype=str)
        colnames = np.loadtxt('%s.colnames.txt' % basename, dtype=str)

        # combine data, row names, and column names into dataframe
        return pd.DataFrame(X, index=rownames, columns=colnames)
    else:
        print('error: filename %s is invalid' % (filename))
        sys.exit(1)



def save_dataframe(filename, df):
    basename, ext = split_filename(filename)

    if ext == 'txt':
        # save dataframe to plaintext file
        df.to_csv(filename, sep='\t', na_rep='NA', float_format='%.8f')
    elif ext == 'npy':
        # save data matrix to binary file
        np.save(filename, df.to_numpy(dtype=np.float32))

        # save row names and column names to text files
        np.savetxt('%s.rownames.txt' % basename, df.index, fmt='%s')
        np.savetxt('%s.colnames.txt' % basename, df.columns, fmt='%s')
    else:
        print('error: filename %s is invalid' % (filename))
        sys.exit(1)



def load_labels(filename, classes=None):
    # load labels file
    df = pd.read_csv(filename, sep='\t', header=None)
    samples, labels = df[0], df[1]

    # infer list of classes if needed
    if classes == None:
        classes = sorted(set(labels))

    # convert categorical labels to numerical labels
    labels = pd.Series(
        index=samples,
        data=[classes.index(l) for l in labels]
    )

    return labels, classes



def load_gene_sets(filename):
    # load file into list
    lines = [line.strip() for line in open(filename, 'r')]
    lines = [re.split(r'[\s,]+', line) for line in lines]

    # map each gene set into a tuple of the name and genes in the set
    return {line[0]: set(line[1:]) for line in lines}



def filter_gene_sets(gene_sets, df_genes):
    # determine the set of genes which are in both
    # the dataset and the list of gene sets
    genes = set().union(*gene_sets.values())
    df_genes = set(df_genes)
    found_genes = genes.intersection(df_genes)

    # remove missing genes from each gene set
    gene_sets = {name: sorted(gene_set.intersection(df_genes)) for name, gene_set in gene_sets.items()}

    print('%d / %d genes from gene sets are in the input dataset' % (len(found_genes), len(genes)))

    return gene_sets



def sanitize(label):
    label = label.replace(' ', '_')
    label = label.replace('.', '_')
    label = label.replace('(', '')
    label = label.replace(')', '')
    return label
