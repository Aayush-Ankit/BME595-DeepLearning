require 'nngraph'
require 'pretty-nn'
torch.manualSeed(0)

-- getNode function for nn.gModule
function nn.gModule:getNode(id)
   for _,n in ipairs (self.forwardnodes) do
      if (n.id == id) then return n.data.module end
   end
   return nil
end

-- make a network using nn package
net = nn.Sequential();
net:add(nn.Linear(20,10));
net:add(nn.Tanh());
net:add(nn.Linear(10,10));
net:add(nn.Tanh());
net:add(nn.Linear(10,1));

-- 1st appraoch to make NN using nngraph
--[[h1 = net.modules[1]()
h2 = net.modules[5](net.modules[4](net.modules[3](net.modules[2](h1))))
gNet = nn.gModule({h1}, {h2})
graph.dot(gNet.fg, 'mlp', 'mlp')--]]

-- 2nd approach to make NN using nnGraph
g1 = - nn.Linear(20,10)
g2 = g1 - nn.Tanh() - nn.Linear(10,10) - nn.Tanh() - nn.Linear(10,1)

mlp = nn.gModule({g1}, {g2})
graph.dot(mlp.fg, 'mlp2', 'mlp2')



