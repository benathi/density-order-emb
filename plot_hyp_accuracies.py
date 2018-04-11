import os
import matplotlib
if os.uname()[0] == 'Linux':
  matplotlib.use('Agg')
from ggplot import *
import pandas as pd
import argparse
import glob
import numpy as np
pd.set_option('display.max_colwidth', -1)

def parse_result(fname, label):
  f = open(fname, 'r')
  _d = {}
  _d['Iteration'] = []
  _d['Y'] = []
  for line in f:
    if line.startswith('Epoch '):
      it = int(line.split('Batch')[1].split('Error')[0])
    elif "Accuracy" in line and "contrastive" in line and "trans" in line:
      acc = float(line.split('Accuracy')[1])
      _d['Iteration'].append(it)
      _d['Y'].append(acc)
    elif False and line.startswith('Mu norm mean') or line.startswith('Mu norm variance') or line.startswith('Sig component mean') or line.startswith('Sig component var'):
      val = float(line.split('\t')[1])
      name = "_".join(line.split('\t')[0].split(' '))
  _d['Label'] = [label]*len(_d['Iteration'])
  df = pd.DataFrame(_d)
  return df

def visualize(pattern_or_list, save=True, labels=None, figname='plot.pdf'):
  # pattern is the pattern for directory / files to parse data and visualize
  if type(pattern_or_list) is str:
    list_files = glob.glob(pattern_or_list)
  elif type(pattern_or_list) is list:
    list_files = pattern_or_list
  dfs = []
  for ii, f in enumerate(list_files):
    basename = os.path.basename(f)# get rid of .log
    if labels is None:
      _df = parse_result(f, label=basename)
    else:
      _df = parse_result(f, label=labels[ii])
    dfs.append(_df)

  df_all = pd.concat(dfs)

  p = (ggplot(aes(x='Iteration', y='Y', color='Label', shape='Label'), data=df_all) 
      + geom_line() + geom_point(aes())
      )
  if save:
    p.save(figname)
  else:
    print p

  return df_all

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--pattern', default='', help='The matching pattern for directories to be plotted. \
    For example, log/exp_vx/test*')
  parser.add_argument('--figname', default=None, help='Figure filename')
  args = parser.parse_args()
  dfs = visualize(args.pattern, save=(args.figname is not None), figname=args.figname)














