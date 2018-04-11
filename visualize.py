"""Usage:
    python visualize.py <results directory>
"""
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt


# metadata of the files to load
DF_META_MAP = {
    'initial': {
        'index.name': 'Game',
        'columns': ['Reward']
    },
    'final': {
        'index.name': 'Game',
        'columns': ['Reward']
    },
    'losses': {
        'index.name': 'Episode',
        'columns': ['Loss']
    },
    'scores': {
        'index.name': 'Episode',
        'columns': ['Reward']
    }
}


# load variables from the command line
try:
    exp_directory = sys.argv[1]
except IndexError:
    print(__doc__)
    sys.exit(-1)

# validate the experiment directory
if not os.path.exists(exp_directory):
    print('{} is not a valid experiments directory'.format(repr(exp_directory)))
    sys.exit(-1)
# create a directory for storing plots
plot_dir = '{}/plots'.format(exp_directory)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


# a mapping of DataFrame names to DataFrames
dfs = {}
# load all the DataFrames
for df_name, df_meta in DF_META_MAP.items():
    file_name = '{}/{}.csv'.format(exp_directory, df_name)
    try:
        df = pd.read_csv(file_name, index_col=0)
        df.index.name = df_meta['index.name']
        df.columns = df_meta['columns']
        dfs[df_name] = df
    except FileNotFoundError:
        print('ERROR: {} not found!'.format(file_name))


# create the training DataFrame and plot it
train = pd.concat([dfs['scores'], dfs['losses']], axis=1)
train.plot(figsize=(12, 5), subplots=True)
plt.savefig('{}/training.pdf'.format(plot_dir))


# plot the results of individual games
for df_name in ['initial', 'final']:
    dfs[df_name].hist()
    plt.savefig('{}/{}.pdf'.format(plot_dir, df_name))
