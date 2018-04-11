local DatasetWSampling = torch.class('DatasetWSampling')
require 'string'
require 'torch'

local function shallow_copy(t)
    local t2 = {}
    for k,v in pairs(t) do
        t2[k] = v
    end
    return t2
end

local function gen_ns(numEntities, descendants, add_parent)
    while true do -- need this loop becauase ds might be an empty list
        while true do -- this one: get an parent index with number of children 2 at least
            idx_p = math.random(numEntities) -- 1 to numEntities
            local num_children = 0
            for k,v in pairs(descendants[idx_p]) do
                num_children = num_children + 1
            end
            -- requires at least 2 children
            if add_parent then
                if num_children > 0 then
                    break
                end
            else
                if num_children > 1 then
                    break
                end
            end
        end
        -- we get desirable idx_p from above
        ds = shallow_copy(descendants[idx_p])
        -- suppose we sample this from ds
        local keyset_ds = {}
        for k in pairs(ds) do
            table.insert(keyset_ds, k)
        end
        idx_child = keyset_ds[math.random(#keyset_ds)]

        ds[idx_child] = nil -- effectively removing from the table

        -- note: if we don't add parent, then we can do two contrast (KL(a,b) and KL(b,a))
        if add_parent then
            ds[idx_p] = true -- add parent to ds: can also contrast with parent if not neighbor
        end
        if next(descendants[idx_child]) ~= nil then
            for kk,vv in pairs(descendants[idx_child]) do
                ds[kk] = nil -- effectively removing from the table
            end
        end
        -- note: ds can be nil if that descendant is the only direct child and we don't add parent
        if add_parent or next(ds) ~= nil then
            local keyset_ds = {}
            for k in pairs(ds) do
                table.insert(keyset_ds, k)
            end
            idx_contrast = keyset_ds[math.random(#keyset_ds)]
            return idx_child, idx_contrast
            -- idx_constrast could be the parent idx
            -- both of these indices should be swapped in the negative sampling?
        end
    end
end

local function gen_neighbor_negatives(num_negatives, numEntities, descendants, add_parent)
    negative_samples = torch.LongTensor(num_negatives, 2)
    for i=1,num_negatives do
        a,b = gen_ns(numEntities, descendants, add_parent)
        negative_samples[i][1] = b
        negative_samples[i][2] = a
    end
    return negative_samples
end

local function genNegatives(N_entities, hypernyms, method, negatives, descendants, opt)
    local negatives = negatives
    if method == 'random' then
        negatives =  torch.rand(hypernyms:size(1), 2):mul(N_entities):ceil():cmax(1):long()
    elseif method == 'contrastive' then
        -- (1) add x portions 'S1' samples (x = num_mult)
        local num_mult = tonumber(opt.s1)
        local randomEntities = torch.rand(hypernyms:size(1)*num_mult, 1):mul(N_entities):ceil():cmax(1):long()
        local index = torch.rand(hypernyms:size(1)*num_mult, 1):mul(2):ceil():cmax(1):long() -- indices, between 1 and 2
        negatives = hypernyms:clone()
        for i=1,(num_mult-1) do
            negatives = torch.cat(negatives, hypernyms, 1)
        end
        negatives:scatter(2, index, randomEntities)
        -- (2) add 1 portions of 'S2'
        -- parent negative
        if opt.s2 == 1 then
            negatives2 = hypernyms:clone()
            negatives2[{{},2}] = hypernyms[{{},1}]
            negatives2[{{},1}] = hypernyms[{{},2}]
            -- contrast with the neighbors
            negatives = torch.cat(negatives, negatives2, 1)
        end
        if opt.s3 == 1 then
            -- (3) 1 portion of S3
            negatives3 = gen_neighbor_negatives(N_entities, N_entities, descendants, true)
            negatives = torch.cat(negatives, negatives3, 1)
        elseif opt.s4 == 1 then
            -- (3) 1 portion of S4
            negatives3 = gen_neighbor_negatives(N_entities, N_entities, descendants, false)
            negatives = torch.cat(negatives, negatives3, 1)
        end
    end
    -- negatives: contain the indicies that would be negative samples
    return negatives
end

-- dataset creation
function DatasetWSampling:__init(N_entities, hypernyms, method, opt, negatives)
    self.method = method
    self.hypernyms = hypernyms
    local N_hypernyms = hypernyms:size(1)

    self:generateDescendants(N_entities, hypernyms)

    self.negativesCounter = 1
    -- define a function with the right parameters that can be used to generate negatives
    self.genNegatives = function() return genNegatives(N_entities, hypernyms, method, negatives, self.descendants, opt) end

    self:regenNegatives()
    self.epoch = 0
end

function DatasetWSampling:generateDescendants(N_entities, hypernyms)
    local descendants = {}
    for ii = 1,N_entities do
        descendants[ii] = {}
    end

    for ii = 1,hypernyms:size(1) do
        hypo = hypernyms[ii][1]
        hyper = hypernyms[ii][2]
        descendants[hyper][hypo] = true
    end
    self.descendants = descendants
end

function DatasetWSampling:regenNegatives()
    local negatives = self.genNegatives()
    -- assuming negatives also has the shape of self.hypernyms 
    local all_hypernyms = torch.cat(self.hypernyms, negatives, 1)
    -- self.hyper and self.hypo contains negative samples too
    -- the first half are true hypernyms, the second half are negative samples
    self.hyper = all_hypernyms[{{}, 2}]
    self.hypo = all_hypernyms[{{}, 1}]
    -- Target specifies if it's actually a correct relationship (1) or incorrect relationship (2)
    self.target = torch.cat(torch.ones(self.hypernyms:size(1)), torch.zeros(negatives:size(1)))
end



function DatasetWSampling:size()
    return self.target:size(1)
end

function DatasetWSampling:minibatch(size)
    if not self.s or self.s + size - 1 > self:size() then
        -- new epoch; randomly shuffle dataset
        self.order = torch.randperm(self:size()):long()
        self.s = 1
        self.epoch = self.epoch + 1
        -- regenerate negatives
        self:regenNegatives()
    end

    local s = self.s
    local e = s + size - 1

    local indices = self.order[{{s,e}}]
    local hyper = self.hyper:index(1, indices)
    local hypo = self.hypo:index(1, indices)
    local target = self.target:index(1, indices)

    self.s = e + 1

    return {hyper, hypo}, target
end

function DatasetWSampling:all()
    return {self.hyper, self.hypo}, self.target
end