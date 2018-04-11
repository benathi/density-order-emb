import numpy as np
import torch

def kl(mu1, mu2, sig1, sig2, epsilon=1e-12):
  # This is KL(1 || 2)
  diff = mu1/np.linalg.norm(mu1) - mu2/np.linalg.norm(mu2)
  sig2_inv = 1./(epsilon + sig2)
  res = np.sum(diff*sig2_inv*diff)
  res += np.sum(sig1*sig2_inv)
  res -= np.sum(np.log(sig1) - np.log(sig2)) # log of ratio of det
  res -= mu1.shape[0] # the dimension
  return res

def elk(mu1, mu2, sig1, sig2, epsilon=1e-12, batch=False):
  if batch:
    mu1_ = mu1[:,None,:]
    sig1_ = sig1[:,None,:]
    mu2_ = mu2[None,:,:]
    sig2_ = sig2[None,:,:]
  else:
    mu1_, mu2_, sig1_, sig2_ = mu1, mu2, sig1, sig2
  assert mu1.shape == mu2.shape, "Shapes do not agree"
  diff = mu1_/np.linalg.norm(mu1_, keepdims=True) - mu2_/np.linalg.norm(mu2_, keepdims=True)
  sig_inv = 1./(epsilon + sig1_ + sig2_)
  res = - np.sum(diff*sig_inv*diff, axis=-1)
  res += - np.sum(np.log(sig1_ + sig2_), axis=-1)
  return -res

def elktorch(mu1, mu2, sig1, sig2, epsilon=1e-12, batch=False):
  if batch:
    mu1_ = mu1[:,None,:]
    sig1_ = sig1[:,None,:]
    mu2_ = mu2[None,:,:]
    sig2_ = sig2[None,:,:]
  else:
    mu1_, mu2_, sig1_, sig2_ = mu1, mu2, sig1, sig2
  assert mu1.shape == mu2.shape, "Shapes do not agree"
  diff = mu1_/mu1_.norm(p=2, dim=-1, keepdim=True) - mu2_/mu2_.norm(p=2, dim=-1, keepdim=True)
  sig_inv = 1./(epsilon + sig1_ + sig2_)
  res = - torch.sum(diff*sig_inv*diff, dim=-1)
  res += - torch.sum(torch.log(sig1_ + sig2_), dim=-1)
  return -res

def kltorch(mu1, mu2, sig1, sig2, epsilon=1e-12, batch=True):
    # This is KL(1 || 2)
    if batch:
      mu1_ = mu1[:,None,:]
      sig1_ = sig1[:,None,:]
      mu2_ = mu2[None,:,:]
      sig2_ = sig2[None,:,:]
    else:
      mu1_, mu2_, sig1_, sig2_ = mu1, mu2, sig1, sig2
    orig_numpy = False
    if type(mu1_).__module__ == np.__name__:
      mu1_ = torch.from_numpy(mu1_)
      mu2_ = torch.from_numpy(mu2_)
      sig1_ = torch.from_numpy(sig1_)
      sig2_ = torch.from_numpy(sig2_)
      orig_numpy = True

    diff = mu1_/mu1_.norm(p=2, dim=-1, keepdim=True) - mu2_/mu2_.norm(p=2, dim=-1, keepdim=True)
    sig2_inv = 1./(epsilon + sig2_)
    res = torch.sum(diff*sig2_inv*diff, dim=-1)
    res += torch.sum(sig1_*sig2_inv, dim=-1)
    res -= torch.sum(torch.log(sig1_) - torch.log(sig2_), dim=-1)
    res -= mu1.shape[-1]
    if orig_numpy:
      return res.cpu().numpy()
    else:
      return res

def kltorch_onetoone(mu1, mu2, sig1, sig2, epsilon=1e-12):
  return kltorch(mu1, mu2, sig1, sig2, epsilon=1e-12, batch=False)

# can do batch (cross mu1 and mu2) or just one-to-one
def alpha_div(mu1, mu2, sig1, sig2, alpha, epsilon=1e-12, batch=False):
  # alpha 1 || 2
  if batch:
    mu1_ = mu1[:,None,:]
    sig1_ = sig1[:,None,:]
    mu2_ = mu2[None,:,:]
    sig2_ = sig2[None,:,:]
  else:
    mu1_, mu2_, sig1_, sig2_ = mu1, mu2, sig1, sig2
  # convert to torch
  orig_numpy = False
  if type(mu1_).__module__ == np.__name__:
    mu1_ = torch.from_numpy(mu1_)
    mu2_ = torch.from_numpy(mu2_)
    sig1_ = torch.from_numpy(sig1_)
    sig2_ = torch.from_numpy(sig2_)
    orig_numpy = True

  assert mu1.shape == mu2.shape, "Shapes do not agree"
  diff = mu1_/mu1_.norm(p=2, dim=-1, keepdim=True) - mu2_/mu2_.norm(p=2, dim=-1, keepdim=True)
  comb_sig = alpha*sig2_ + (1-alpha)*sig1_
  res = torch.sum(diff*(1/(epsilon + torch.abs(comb_sig)))*diff, dim=-1)
  res += -(1/(alpha*(alpha-1)))*torch.sum(torch.log(torch.abs(comb_sig)), dim=-1)
  res += -(1/alpha)*torch.sum(torch.log(sig1_), dim=-1)
  res += -(1/(1-alpha))*torch.sum(torch.log(sig2_), dim=-1)
  if orig_numpy:
    return res.cpu().numpy()
  else:
    return res

def divergence(mu1, mu2, sig1, sig2, alpha=1, epsilon=1e-12, cross_batch=False):
  if cross_batch:
    mu1_ = mu1[:,None,:]
    sig1_ = sig1[:,None,:]
    mu2_ = mu2[None,:,:]
    sig2_ = sig2[None,:,:]
  else:
    mu1_, mu2_, sig1_, sig2_ = mu1, mu2, sig1, sig2
  if alpha == 1:
    return kltorch(mu1, mu2, sig1, sig2, epsilon=epsilon, batch=cross_batch)
  else:
    return alpha_div(mu1, mu2, sig1, sig2, alpha=alpha, epsilon=epsilon, batch=cross_batch)