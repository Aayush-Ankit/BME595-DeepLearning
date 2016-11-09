require 'nn'
require 'pretty-nn'
require 'nngraph'

torch.manualSeed(0)

-- getNode function for nn.gModule
function nn.gModule:getNode(id)
   for _,n in ipairs (self.forwardnodes) do
      if (n.id == id) then return n.data.module end
   end
   return nil
end

-- create a fancy network using nngraph
input = - nn.Identity()
L1 = input - nn.Linear(10, 20) - nn.Tanh()
L2 = {input, L1} - nn.JoinTable(1) - nn.Linear(30,60) - nn.Tanh()
L3 = {L1, L2} - nn.JoinTable(1) - nn.Linear(80,1) - nn.Tanh()

g = nn.gModule({input}, {L3})
graph.dot (g.fg, 'fancy', 'fancy')

