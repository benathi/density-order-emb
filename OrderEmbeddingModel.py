import numpy as np
from eval.evalutil import kl, kltorch
import os
import torch
from torch.utils.serialization import load_lua

class DensityOrderEmbedding:
  def __init__(self, dirpath="modelfiles/model_hyp_1_margin2000.0_var0.00005_klthres500_init_norm/",
      namepath="dataset/synset_names.txt", verbose=False,
      use_torch=True, use_cuda=True):
    # all the paths are with respect to the file location
    cur_abspath = os.path.abspath(os.path.dirname(__file__))
    obj_mus = load_lua(os.path.join(cur_abspath, dirpath, "mus.t7"), unknown_classes=False)
    mus = obj_mus.weight.numpy()
    obj_logsigs = load_lua(os.path.join(cur_abspath, dirpath, "logsigs.t7"), unknown_classes=False)
    logsigs = obj_logsigs.weight.numpy()
    id2word = []
    _namepath = os.path.join(cur_abspath, namepath)
    f = open(_namepath)
    for i,line in enumerate(f):
      word = line.strip()
      id2word.append(word)
    if verbose: print "Initializing"
    self.dim = mus.shape[1]
    sigs = np.exp(logsigs)
    self.id2synset = id2word
    self.num_entities = len(id2word)
    self.synset2id = {}
    for idx, synset in enumerate(self.id2synset):
      self.synset2id[synset] = idx
    self.word2idxs = {}
    for idx, synset in enumerate(self.id2synset):
      word = ".".join(synset.split(".")[:-2])
      if word not in self.word2idxs:
        self.word2idxs[word] = [idx]
      else:
        self.word2idxs[word].append(idx)
    self.mu = mus
    self.sig = sigs
    if verbose >= 2: print "Initializing KL table"
    if verbose >= 2: print "Done initializing KL table"
    # populate torch vector
    self.use_cuda = use_cuda
    if use_torch:
      self.tmus = torch.from_numpy(mus)
      self.tsigs = torch.from_numpy(sigs)
      if use_cuda:
        self.tmus = self.tmus.type(torch.cuda.FloatTensor)
        self.tsigs = self.tsigs.type(torch.cuda.FloatTensor)

  def get_idxs(self, word):
    return self.word2idxs[word]

  def get_synsets(self, word):
    idxs = self.get_idxs(word)
    result = []
    for idx in idxs:
      result.append(self.id2synset[idx])
    return result

  def get_idx(self, word):
    return self.get_idxs(word)[0] # the first element (probably more common)

  def kl(self, syn1, syn2):
    klval = None
    if type(syn1) is str:
      idx1 = self.synset2id[syn1]
      idx2 = self.synset2id[syn2]
    else:
      idx1, idx2 = int(syn1), int(syn2)
    klval = kl(self.mu[idx1], self.mu[idx2], self.sig[idx1], self.sig[idx2])
    return klval

  def kl_batch(self, syn1, syn2):
    assert len(syn1) >= 1 and len(syn2) >= 1, "syn1 and syn2 must be lists or arrays"
    syn1_ = torch.LongTensor(syn1)
    syn2_ = torch.LongTensor(syn2)
    if self.use_cuda:
      syn1_ = syn1_.type(torch.cuda.LongTensor)
      syn2_ = syn2_.type(torch.cuda.LongTensor)
    return kltorch(self.tmus[syn1_], self.tmus[syn2_], self.tsigs[syn1_], self.tsigs[syn2_])

if __name__ == "__main__":
  c = DensityOrderEmbedding(verbose=2)