import copy
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sys
import re



def split_filename(filename):
    tokens = filename.split('.')

    return '.'.join(tokens[:-1]), tokens[-1]



def load_dataframe(filename):
    basename, ext = split_filename(filename)

    if ext == 'txt':
        # load dataframe from plaintext file
        return pd.read_csv(filename, index_col=0, sep='\t')
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
        np.save(filename, np.array(df.values, dtype=np.float32, order='F'))

        # save row names and column names to text files
        np.savetxt('%s.rownames.txt' % basename, df.index, fmt='%s')
        np.savetxt('%s.colnames.txt' % basename, df.columns, fmt='%s')
    else:
        print('error: filename %s is invalid' % (filename))
        sys.exit(1)



def load_labels(filename, classes=None):
    # load labels file
    labels = pd.read_csv(filename, sep='\t', header=None, index_col=0)
    labels = labels[1].values

    # convert categorical labels to numerical labels
    if classes != None:
        labels = np.array([classes.index(l) for l in labels])
    else:
        encoder = sklearn.preprocessing.LabelEncoder()
        labels = encoder.fit_transform(labels)
        classes = list(encoder.classes_)

    return labels, classes



def load_gene_sets(filename):
    # load file into list
    lines = [line.strip() for line in open(filename, 'r')]
    lines = [re.split(r'[\s,]+', line) for line in lines]

    # map each gene set into a tuple of the name and genes in the set
    return {line[0]: line[1:] for line in lines}



def filter_gene_sets(gene_sets, df_genes):
    # compute the union of all gene sets
    genes = list(set(sum([gene_sets[name] for name in gene_sets.keys()], [])))

    # determine the genes which are not in the dataset
    missing_genes = [g for g in genes if g not in df_genes]

    # remove missing genes from each gene set
    gene_sets = {name: [g for g in genes if g in df_genes] for name, genes in gene_sets.items()}

    print('%d / %d genes from gene sets were not found in the input dataset' % (len(missing_genes), len(genes)))

    return gene_sets



def onehot_encode(y, classes):
    return np.eye(len(classes))[y]



def shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    return x[indices], y[indices]



def next_batch(x, y, batch_size, index):
    a = index * batch_size
    b = index * batch_size + batch_size

    return x[a:b], y[a:b]



def sanitize(label):
    label = label.replace(' ', '_')
    label = label.replace('.', '_')
    label = label.replace('(', '')
    label = label.replace(')', '')
    return label
