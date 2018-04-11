require 'nn'
require 'dpnn'
require 'nngraph'

local f = {}
-- alpha divergence
function f.HypernymScoreGaussAlpha(params, num_entities)
    print('Using HypernymScoreGaussAlpha (Alpha Divergence) with alpha =' .. tostring(params.alpha))
    local input_1 = nn.Identity()()
    local input_2 = nn.Identity()()

    local mus = nn.LookupTable(num_entities, params.D_embedding)
    local embedding_mus = nn.Sequential():add(mus)
    local embedding_mus2 = embedding_mus:sharedClone()

    -- D_\alpha(D_2 || D_1)
    -- D_\alpha(f || g) -> g=1, f=2
    -- D_1 is supposed to be the hypernym
    local alpha = params.alpha

    local logsigs = nn.LookupTable(num_entities, params.D_embedding)
    local embedding_logsigs = nn.Sequential():add(logsigs)
    local embedding_logsigs2 = embedding_logsigs:sharedClone()

    if params.normalize then
        print('Normalize the embeddings')
        embedding_mus:add(nn.Normalize(2))
        embedding_mus2:add(nn.Normalize(2))
    end
    
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
    -- note: need the embeddings to be different objects (embeddings_mus and embeddings_mus2)
    -- but share parameters.

    -- This is the scaled sum part
    local mu_diff = nn.CSubTable()({mus_1, mus_2})
    local mu_diff2 = nn.CSubTable()({mus_1, mus_2})
    local comb_sig = nn.CAddTable()({nn.MulConstant(alpha, false)(nn.Exp()(logsigs_1)), nn.MulConstant(1-alpha, false)(nn.Exp()(logsigs_2))})
    -- should add jitter to make it stable
    local inv_sig1 = nn.Power(-1)(comb_sig)
    local scaled_dot = nn.CMulTable()({mu_diff, inv_sig1, mu_diff2})
    local scaled_dot_sum = nn.Sum(2)(scaled_dot)

    local logdet_term1 = nn.MulConstant(-1./(alpha*(alpha-1)),false)(nn.Sum(2)(nn.Log()(nn.Abs()(comb_sig))))
    local logdet_term2 = nn.MulConstant(-1./alpha,false)(nn.Sum(2)(logsigs_2))
    local logdet_term3 = nn.MulConstant(1./(alpha-1))(nn.Sum(2)(logsigs_1))
    
    local out = nn.CAddTable()({scaled_dot_sum, logdet_term1, logdet_term2, logdet_term3})

    local out3 = out

    if USE_CUDA then
        print('Sharing parameter again')
        embedding_mus:share(embedding_mus2, 'weight', 'bias', 'gradWeight', 'gradBias')
        embedding_logsigs:share(embedding_logsigs2, 'weight', 'bias', 'gradWeight', 'gradBias')
    end

    return nn.gModule({input_1, input_2}, {out3}), mus, logsigs
end

return f




















