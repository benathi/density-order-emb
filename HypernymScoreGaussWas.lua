--require 'cutorch'
require 'nn'
--require 'cunn'
require 'dpnn'
require 'nngraph'

--local HypernymScoreGauss, parent = torch.class('nn.HypernymScoreGauss', 'nn.Sequential')
local f = {}

function f.hypernymScoreGaussWas(params, num_entities)
--function HypernymScoreGaussG:__init(params, num_entities)
    print('in hypernymScoreGaussWas')
    --parent.__init(self)
    local input_1 = nn.Identity()()
    local input_2 = nn.Identity()()

    local mus = nn.LookupTable(num_entities, params.D_embedding) -- was lookup before
    local embedding_mus = nn.Sequential():add(mus)
    local embedding_mus2 = embedding_mus:sharedClone()

    -- KL(D_2 || D_1)
    -- D_1 is supposed to be the hypernym? - we can reverse this to test I guess
    -- The vector loss is D_1 - D_2 which implies that the more general entity should be D_1
    -- to reduce this loss

    local logsigs = nn.LookupTable(num_entities, params.D_embedding)
    local embedding_logsigs = nn.Sequential():add(logsigs)
    local embedding_logsigs2 = embedding_logsigs:sharedClone()

    if params.normalize then
        print('Normalize the embeddings')
        embedding_mus:add(nn.Normalize(2))
        embedding_mus2:add(nn.Normalize(2))
    end
    
    -- switching would not matter because it's symmetric
    local mus_1, logsigs_1, mus_2, logsigs_2
    if params.hyp == '1' then
      mus_1 = embedding_mus(input_1)
      logsigs_1 = embedding_logsigs(input_1)
      mus_2 = embedding_mus2(input_2)
      logsigs_2 = embedding_logsigs2(input_2)
    elseif params.hyp == '2' then
      mus_1 = embedding_mus(input_2)
      logsigs_1 = embedding_logsigs(input_2)
      mus_2 = embedding_mus2(input_1)
      logsigs_2 = embedding_logsigs2(input_1)
    end

    local mu_diff = nn.CSubTable()({mus_1, mus_2})
    local scaled_dot_sum = nn.Sum(2)(nn.Power(2)(mu_diff))
    
    local var_diff = nn.CSubTable()({ nn.Power(0.5)(nn.Exp()(logsigs_1)) , nn.Power(0.5)(nn.Exp()(logsigs_2)) })
    local trace_term = nn.Sum(2)(nn.Power(2)(var_diff))
    local out = nn.CAddTable()({scaled_dot_sum, trace_term})

    print('Wasserstein threshold = ' .. params.kl_threshold)
    local out2 = nn.AddConstant(-params.kl_threshold, false)(out)
    local out3 = nn.ReLU()(out2)

    if USE_CUDA then
        print('Sharing parameter again')
        embedding_mus:share(embedding_mus2, 'weight', 'bias', 'gradWeight', 'gradBias')
        embedding_logsigs:share(embedding_logsigs2, 'weight', 'bias', 'gradWeight', 'gradBias')
    end

    return nn.gModule({input_1, input_2}, {out3}), mus, logsigs
end

return f