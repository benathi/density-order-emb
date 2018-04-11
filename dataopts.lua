local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Dataset creation - WordNet Hypernym')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-s1',                 1,    'Number of layers in block 1')
   cmd:option('-s2',                 0,    'Number of layers in block 2')
   cmd:option('-s3',                 0,    'Number of layers in block 3')
   cmd:option('-s4',                 0,    'Number of layers in block 4')

   cmd:text()

   local opt = cmd:parse(arg or {})
   return opt
end

return M