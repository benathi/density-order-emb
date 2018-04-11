--require 'Dataset'
require 'DatasetWSampling'
torch.manualSeed(1234)
local opts = require 'dataopts'
local sampling_opt = opts.parse(arg)
local method = 'contrastive'
print("Negative Sampling Options")
print(sampling_opt)
local hdf5 = require 'hdf5'

local f = hdf5.open('dataset/wordnet.h5', 'r')
local originalHypernyms = f:read('hypernyms'):all():add(1) -- convert to 1-based indexing
local numEntities = torch.max(originalHypernyms) 
f:close()
print("Loaded data")

local graph = require 'Graph'

-----
-- split hypernyms into train, dev, test
-----
for _, hypernymType in ipairs{'trans', 'notrans'} do
    local methodName = method
    local hypernyms = originalHypernyms
    if hypernymType == 'trans' then
        hypernyms = graph.transitiveClosure(hypernyms)
        methodName = methodName .. '_trans'
        print("Creating dataset " .. methodName)
    end

    local N_hypernyms = hypernyms:size(1)
    print("Total number of hypernym relationships = " .. N_hypernyms)
    -- trans: 837888
    -- notrans: 84427
    local splitSize = 4000

    -- shuffle randomly
    torch.manualSeed(1)
    local order = torch.randperm(N_hypernyms):long()
    local hypernyms = hypernyms:index(1, order)
    print("Building sets ...")

    local sets = {
            test = hypernyms:narrow(1, 1, splitSize),
            val = hypernyms:narrow(1, splitSize + 1, splitSize),
            train = hypernyms:narrow(1, splitSize*2+ 1, N_hypernyms - 2*splitSize)
        }
    print("Done. Building Datasets ...")
    local datasets = {}
    for name, hnyms in pairs(sets) do
        datasets[name] = DatasetWSampling(numEntities, hnyms, method, sampling_opt)
    end

    datasets.numEntities = numEntities

    -- save visualization info
    local paths = require 'paths'
    local json = require 'cjson'
    local function write_json(file, t)
        local filename = file .. '.json'
        paths.mkdir(paths.dirname(filename))
        local f = io.open(filename, 'w')
    end

    local fname = methodName
    fname = fname .. '_s1-' .. tostring(sampling_opt.s1) 
    if sampling_opt.s2 ~= 0 then
        fname = fname .. '_s2-' .. tostring(sampling_opt.s2) 
    end
    if sampling_opt.s3 ~= 0 then
        fname = fname .. '_s3-' .. tostring(sampling_opt.s3) 
    end
    if sampling_opt.s4 ~= 0 then
        fname = fname .. '_s4-' .. tostring(sampling_opt.s4) 
    end
    local datapath = 'dataset/' .. fname .. '.t7'
    print("Saving data to" .. datapath)
    torch.save(datapath, datasets)
end