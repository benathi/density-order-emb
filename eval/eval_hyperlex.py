# Evaluating hyperlex data using the model
import scipy.stats
import numpy as np
import sys
sys.path.append("../")
sys.path.append(".")
import OrderEmbeddingModel
import argparse
from evalutil import divergence
import os
from ggplot import *
import pandas as pd

def load_hyperlex(hyperlex_path="hyperlex-data/nouns-verbs/hyperlex-nouns.txt"):
  _path = os.path.join(os.path.abspath(os.path.dirname(__file__)), hyperlex_path)
  f = open(_path)
  words = []
  hyps = []
  scores6 = []  # full score = 6
  scores10 = [] # full score = 10
  reltypes = [] # to be displayed for debugging
  for i, line in enumerate(f):
    if i == 0: continue # ignore the header
    ll = line.split()
    word, hyp, score6, score10, reltype = [ll[0], ll[1], float(ll[4]), float(ll[5]), ll[3]]
    words.append(word)
    hyps.append(hyp)
    scores6.append(score6)
    scores10.append(score10)
    reltypes.append(reltype)
  return words, hyps, scores6, scores10, reltypes

def elk(mu1, mu2, sig1, sig2):
  _a = sig1 + sig2 # Is this stable? maybe store logsig instead?
  _res = -0.5*np.sum(np.log(_a))
  ss_inv = 1./_a
  diff = mu1 - mu2
  _res += -0.5*np.sum(diff*ss_inv*diff)
  return _res

def calculate_model_scores(model, words, hyps, metric='dot', alpha=1, verbose=False, real_scores=None, num_shown=None, reltypes=None):
  assert metric  in ['dot', 'kl', 'reversekl', 'elk'], "Please Choose valid metrics"
  scores = []
  exists = []
  for jj, (word, hyp) in enumerate(zip(words, hyps)):
    if word in model.word2idxs and hyp in model.word2idxs:
      exist = True
      idx1, idx2 = [model.get_idx(word), model.get_idx(hyp)]
      if metric == 'dot':
        score = np.dot(model.mu[idx1], model.mu[idx2])
      elif metric == 'cosine':
        score = 1 # find the scipy function
      elif metric == 'kl':
        # negative kl so that the high loss (not true hypernym) corresponds to
        # Take maximum over all the KL pairs
        kl_scores = []
        for idx1 in model.get_idxs(word):
          for idx2 in model.get_idxs(hyp):
            kl_score = divergence(model.mu[idx1], model.mu[idx2], model.sig[idx1], model.sig[idx2], alpha=alpha, cross_batch=False)
            kl_scores.append(kl_score)
        min_kl = min(kl_scores)
        score = - min_kl
      elif metric == 'reversekl':
        kl_scores = []
        for idx1 in model.get_idxs(word):
          for idx2 in model.get_idxs(hyp):
            kl_score = divergence(model.mu[idx2], model.mu[idx1], model.sig[idx2], model.sig[idx1], alpha=alpha, cross_batch=False)
            kl_scores.append(kl_score)
        min_kl = min(kl_scores)
        score = - min_kl
      elif metric == 'elk':
        score = elk(model.mu[idx1], model.mu[idx2], model.sig[idx1], model.sig[idx2])
      if verbose >=2:
        if num_shown is None or jj < num_shown:
          print "{} & {} [t={}, s={}] Score ({}) = {}".format(word,
            hyp, reltypes[jj], real_scores[jj], metric, score)
    else:
      exist = False
      if verbose >=2 and (num_shown is None or jj < num_shown):
        print "######"
        if word not in model.word2idxs:
          print "\t{} NOT in model for pair ({},{})".format(word, word, hyp)
        if hyp not in model.word2idxs:
          print "\t{} NOT in model for pair ({},{})".format(hyp, word, hyp)
      score = 0.0 # note: this is set in calculate_median_scores
    scores.append(score)
    exists.append(exist)
  return scores, exists

def calculate_median_scores(scores, exists, verbose=0):
  # scores: the model scores in list
  # exists: the list containing binary values indicating whether the pair is found in the model
  scores_exist_only = []
  for ii, (score, exist) in enumerate(zip(scores, exists)):
    if exist:
      scores_exist_only.append(score)
  median_score = np.median(np.array(scores_exist_only))
  if verbose > 1: print "The median score is ", median_score
  for ii, exist in enumerate(exists):
    if not exist:
      scores[ii] = median_score
  if verbose > 1: print "Done setting the median score to", median_score
  return median_score

def test_metrics():
  # deprecated
  model = OrderEmbeddingModel.DensityOrderEmbedding(path="dataset/bestmodel_v1.results")
  # test the metric value for a set of synsets
  # suppose we know the synsets!
  list_syns = ['city.n.01', 'location.n.01', 'living_thing.n.01', 'whole.n.02', 'object.n.01', 'physical_entity.n.01']
  for syn in list_syns:
    for syn2 in list_syns:
      idx1, idx2 = model.synset2id[syn], model.synset2id[syn2]
      score = kl(model.mu[idx1], model.mu[idx2], model.sig[idx1], model.sig[idx2])
      print "Syn {} Syn2 {} KL {}".format(syn, syn2, score)


def evaluate_spearmann(modeldir, metrics=['kl'], alpha=1, verbose=False, plot=False, num_shown=10):
  words, hyps, scores6, scores10, reltypes = load_hyperlex()
  if type(modeldir) is str:
    model = OrderEmbeddingModel.DensityOrderEmbedding(dirpath=modeldir)
  else:
    model = modeldir
  spr_corrs = []
  for metric in metrics:
    model_scores, exists = calculate_model_scores(model, words, hyps, metric=metric, alpha=alpha, verbose=verbose, real_scores=scores10, num_shown=num_shown, reltypes=reltypes)
    calculate_median_scores(model_scores, exists, verbose=verbose)
    if verbose >= 2:
      # for each pair, print the model score and the human score
      pass
    if plot:
      ms_list = []
      rs_list = []
      wlist = []
      tlist = []
      for jj, (word, hyp) in enumerate(zip(words, hyps)):
        exist = exists[jj]
        if exist:
          ms = model_scores[jj]
          rs = scores10[jj]
          ms_list.append(ms)
          rs_list.append(rs)
          wlist.append("{}/{}".format(word, hyp))
          reltype = reltypes[jj] # collapse hyp-1, hyp-2, etc together
          if "1" in reltype or "2" in reltype or "3" in reltype or "4" in reltype:
            reltype = reltype[:-2]
          tlist.append(reltype)
        if num_shown is not None and jj >= num_shown:
          break
      _d = {}
      _d['real_score'] = rs_list
      _d['model_score'] = ms_list
      _d['label'] = wlist
      _d['type'] = tlist
      df = pd.DataFrame(_d)
      # how to make the plot show a different scale? - TODO
      print (ggplot(aes(x='real_score', y='model_score', label='label', color='type'), data=df) 
         + geom_point(aes()) + geom_text(aes())
        )

    spr = scipy.stats.spearmanr(scores6, model_scores)
    if verbose:
      print 'Metric {} Spearman correlation is {} with pvalue {}'.format(metric, spr.correlation, spr.pvalue)
    spr_correlation = spr.correlation
    spr_corrs.append(spr_correlation)
  return spr_corrs

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--verbose', type=int, default=0, help='The levels of verbosity. 0-2')
  parser.add_argument('--plot', type=int, default=0, help='whether to plot')
  parser.add_argument('--num', type=int, default=10, help='the number of examples to be shown')
  parser.add_argument('--unknown', type=float, default=-300, help='The score for unknown')
  parser.add_argument('--modeldir', type=str,
    default="modelfiles/model_hyp_1_margin2000.0_var0.00005_klthres500_init_norm/")
  args = parser.parse_args()
  evaluate_spearmann(modeldir=args.modeldir, metrics=['kl', 'dot', 'elk', 'reversekl'],verbose=args.verbose, plot=args.plot, num_shown=args.num)

