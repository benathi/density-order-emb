local argparse = require 'argparse'

parser = argparse('Train a Gaussian order embedding model')
parser:option '--seed' :description 'random seed' : default '1234' :convert(tonumber)
parser:option '-d' :description 'dimensionality of embedding space' :default "50" :convert(tonumber)
parser:option '--epochs' :description 'number of epochs to train for ' :default "1" :convert(tonumber)
parser:option '--batchsize' :description 'size of minibatch to use' :default "1000" :convert(tonumber)
parser:option '--eval_freq' :description 'evaluation frequency' :default "100" :convert(tonumber)
parser:option '--lr' :description 'learning rate' :default "0.01" :convert(tonumber)
parser:option '--train' :description 'dataset to use for training' :default 'contrastive_trans_s1-1'
parser:option '--eval' :description 'dataset to use for evaluation' :args('*')
parser:option '--name' :description 'name of model' :default 'anon'
parser:option '--margin' :description 'size of margin to use for contrastive learning' :default '1' :convert(tonumber)
parser:option '--norm' :description 'norm to use, "inf" for infinity' :default "2" :convert(tonumber)
parser:option '--eps' :description 'constant to use to prevent hypernyms from being mapped to the exact same spot' :default "0" :convert(tonumber)
parser:flag '--symmetric' : description 'use symmetric dot-product distance instance'
parser:option '--rep' :description 'What representation to use, vec or gauss' :default 'gauss'
parser:flag '--init_normalize' :description 'whether to normalize at the beginning' -- true or false
parser:flag '--normalize' :description 'whether to normalize in the network' -- true or false
parser:option '--varscale' :description 'the variance scale' :default '0.05' :convert(tonumber)
parser:option '--kl_threshold' :description 'The threshold for KL below which the loss is zero' :default '0.1' :convert(tonumber)
parser:option '--hyp' :description 'Which one of the input is the hypernym. 1 for usual setting. 2 for reverse.' : default '1' 
parser:option '--aggregate_kl' :description 'How to aggregate the KL of other components (min, max, mean) ' : default 'min'
parser:option '--energy' :description 'KL or ELK' : default 'kl'
parser:option '--spherical' :description 'whether to use spherical' : default 'false' -- true or false
parser:option '--alpha' :description 'alpha for the alpha divergence' : default '1' :convert(tonumber)  -- by default, alpha == 1 for KL
parser:option '--negativealpha': description 'for negative alpha' : default '0' : convert(tonumber)
parser:option '--wasserstein': description 'whether to use Wasserstein' : default '0' : convert(tonumber)


USE_CUDA = true
if USE_CUDA then
    require 'cutorch'
    require 'cunn'
end

local args = parser:parse()
if not args.eval then
    args.eval = {args.train}
end
torch.manualSeed(args.seed)

print(args)

require 'DatasetWSampling'
local datasets = torch.load('dataset/' .. args.train .. '.t7')
local datasets_eval = {}
--The default argument for eval is * -> using the same set for evaluation
print("Using Dataset for evaluation")
print(args.eval)
for _, name in ipairs(args.eval) do
    print('Dataset for evaluation: dataset/' .. name .. '.t7')
    datasets_eval[name] = torch.load('dataset/' .. name .. '.t7')
end
local train = datasets.train


if args.negativealpha == 1 then
  args.alpha = -args.alpha
end

local hyperparams = {
    D_embedding = args.d,
    symmetric = args.symmetric,
    margin = args.margin,
    lr = args.lr,
    norm = math.min(9999, args.norm),
    eps = args.eps,
    kl_threshold = args.kl_threshold,
    normalize = args.normalize,
    hyp = args.hyp,
    aggregate_kl = args.aggregate_kl,
    spherical = args.spherical, -- this might cause some problem
    alpha = args.alpha
}
print('creating timestamp')
local timestampedName = os.date("%Y-%m-%d_%H-%M-%S") .. "_" .. args.name

require 'optim'
require 'HypernymScore'
local HypernymScoreGaussG = require('HypernymScoreGaussG.lua')
local hypernymScoreGaussGELK = require('HypernymScoreGaussGELK.lua')
local HypernymScoreGaussAlpha = require('HypernymScoreGaussAlpha.lua')
local hypernymScoreGaussWas = require('HypernymScoreGaussWas.lua')
local config = { learningRate = args.lr }

local hypernymNet, criterion 
local embed_mus, embed_logsigs 
local best_embed_mus, best_embed_logsigs
if args.rep == 'vec' then
  print('Num entities =') -- 82115
  print(datasets.numEntities)
  hypernymNet = nn.HypernymScore(hyperparams, datasets.numEntities)
  criterion = nn.HingeEmbeddingCriterion(args.margin)
elseif args.rep == 'gauss' then
  print('Building Hypernym Net for Gauss Rep')
  hypernymNet, embed_mus, embed_logsigs = nil

  if args.wasserstein == 0 then
    if args.alpha == 1 then
      if args.energy == 'kl' or args.energy == nil then
        print('Using KL Energy')
        hypernymNet, embed_mus, embed_logsigs = HypernymScoreGaussG.HypernymScoreGaussG(hyperparams, datasets.numEntities)
      elseif args.energy == 'elk' then
        print('Using negative log ELK Energy')
        print(hypernymScoreGaussGELK)
        hypernymNet, embed_mus, embed_logsigs = hypernymScoreGaussGELK.hypernymScoreGaussGELK(hyperparams, datasets.numEntities)
      end
    else
      -- for alpha is not 1: non KL case
      -- make sure alpha is not 0 either
      print("Using Renyi Divergence Model with Alpha =" .. tostring(args.alpha))
      hypernymNet, embed_mus, embed_logsigs = HypernymScoreGaussAlpha.HypernymScoreGaussAlpha(hyperparams, datasets.numEntities)
    end
  else
    print("Using Wasserstein Distance")
    hypernymNet, embed_mus, embed_logsigs = hypernymScoreGaussWas.hypernymScoreGaussWas(hyperparams, datasets.numEntities)
  end

  hypernymNet = hypernymNet:cuda()
  print('Building Criterion')
  criterion = nn.HingeEmbeddingCriterion(args.margin)
end

----------------
-- EVALUATION --
----------------

local function cudify(input, target)
    if USE_CUDA then
        return {input[1]:cuda(), input[2]:cuda()}, target
    else
        return input, target
    end
end

-- returns optimal threshold, and classification at that threshold, for the given dataset
local function findOptimalThreshold(dataset, model)
    local input, target = cudify(dataset:all())
    local probs = model:forward(input):double()
    local sortedProbs, indices = torch.sort(probs, 1) -- sort in ascending order
    local sortedTarget = target:index(1, indices:long())
    local tp = torch.cumsum(sortedTarget)
    local invSortedTarget = torch.eq(sortedTarget, 0):double()
    local Nneg = invSortedTarget:sum() -- number of negatives
    local fp = torch.cumsum(invSortedTarget)
    local tn = fp:mul(-1):add(Nneg)
    local accuracies = torch.add(tp,tn):div(sortedTarget:size(1))
    local bestAccuracy, i = torch.max(accuracies, 1)
    print("Number of positives, negatives, tp, tn: " .. target:sum() .. ' ' .. Nneg .. ' ' .. tp[i[1]] .. ' ' .. tn[i[1]] )
    return sortedProbs[i[1]], bestAccuracy[1]
end

-- evaluate model at given threshold
local function evalClassification(dataset, model, threshold)
    local input, target = cudify(dataset:all())
    local probs = model:forward(input):double()

    local inferred = probs:le(threshold)
    local accuracy = inferred:eq(target:byte()):double():mean()
    return accuracy
end

--------------
-- TRAINING --
--------------
local parameters, gradients = hypernymNet:getParameters()

local best_accuracies = {}
local best_counts = {}
local saved_weight
local count = 1

-- initialize
if args.rep == 'gauss' and args.init_normalize then
  print('Initialize Mus with Unit Norm')
  embed_mus.weight:copy(torch.cdiv(embed_mus.weight, embed_mus.weight:norm(2,2):expandAs(embed_mus.weight))) -- 2 norm over 2nd axis
  print('Using varscale = ' .. args.varscale)
  embed_logsigs.weight:copy(torch.add(torch.mul(embed_logsigs.weight, 0.0), torch.log(args.varscale)))
end

while train.epoch <= args.epochs do
    count = count + 1
    if not args.symmetric and args.rep =='vec' then
        local weight = hypernymNet.lookupModule.weight
        weight:cmax(0) -- make sure weights are positive
    end

    local function eval(x)
        hypernymNet:zeroGradParameters()
        local input, target = cudify(train:minibatch(args.batchsize))
        target:mul(2):add(-1) -- convert from 1/0 to 1/-1 convention
        local probs = hypernymNet:forward(input):double()
        local err = criterion:forward(probs, target)
        if count % 10 == 0 then
            print("Epoch " .. train.epoch .. " Batch " .. count .. " Error " .. err)
        end
        local gProbs = criterion:backward(probs, target):cuda()
        local _ = hypernymNet:backward(input, gProbs)
        return err, gradients
    end

    optim.adam(eval, parameters, config)

    if count % args.eval_freq == 0 then
        for name, dataset in pairs(datasets_eval) do
            local threshold, accuracy = findOptimalThreshold(dataset.val, hypernymNet)
            print("Best accuracy " .. accuracy .. " at threshold " .. threshold) -- This is validation accuracy
            local real_accuracy = evalClassification(dataset.test, hypernymNet, threshold)
            print(name .. " Accuracy " .. real_accuracy) -- this is the test accuracy
            if not best_accuracies[name] or real_accuracy > best_accuracies[name] then
                best_accuracies[name] = real_accuracy
                best_counts[name] = count
                if args.rep == 'vec' then
                  saved_weight = hypernymNet.lookupModule.weight:float()
                end

                best_embed_mus = embed_mus:clone()
                best_embed_logsigs = embed_logsigs:clone()
            end
        end

        if args.rep == 'gauss' then
          print('Evaluating Mus and Logsigs')
          for name, dataset in pairs(datasets_eval) do
            local input, target = cudify(dataset.val:all())
            --print(input) {words, hypernyms}
            --print(target) a vector of 0 or 1
            mus_val = embed_mus:forward(input[1])
            sigs_val = torch.exp(embed_logsigs:forward(input[1]))
            mu_norms = mus_val:norm(2,2) -- 2 norm over 2nd axis
            local norm_mean = mu_norms:mean()
            if norm_mean ~= norm_mean then
              print("NaN Detected!")
              return
            end
            print('Mu norm mean\t' .. mu_norms:mean()) -- vector size of dataset.val
            print('Mu norm variance\t' .. mu_norms:std())
            print('Sig component mean\t' .. sigs_val:mean())
            print('Sig component var\t' .. sigs_val:std())
          end
        end
    end
end

local pretty = require 'pl.pretty'
print("Best accuracy was at batch #" )
print(pretty.write(best_counts,""))
print(pretty.write(best_accuracies,""))

if args.rep == 'vec' then
  torch.save('weights.t7', saved_weight)
end

if args.rep == 'gauss' then
  print('Saving Gauss model variables')
  paths.mkdir(paths.concat('modelfiles', args.name))
  torch.save(paths.concat('modelfiles', args.name, 'mus.t7'), best_embed_mus:float())
  torch.save(paths.concat('modelfiles', args.name, 'logsigs.t7'), best_embed_logsigs:float())
end
