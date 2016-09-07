--This is the API for logic gates
torch.setdefaulttensortype('torch.FloatTensor')
local nn = require 'NeuralNetwork'

local logicGates =  {}
local theta
function logicGates.AND(x,y)

   --convert boolean to float
   local in_vec = torch.Tensor(2,1)
   in_vec[1]  = (x == true) and 1 or 0
   in_vec[2]  = (y == true) and 1 or 0

   --nn.build(torch.Tensor({2,1}))
   nn.build({2,1})
   theta = nn.getLayer(1)
   --theta[1] = torch.Tensor({{20,20,-30}}):t()
   theta[1] = 20
   theta[2] = 20
   theta[3] = -30
   return (((nn.forward (in_vec))[1][1]) > 0.5)
end

function logicGates.OR(x,y)

   --convert boolean to float
   local in_vec = torch.Tensor(2,1)
   in_vec[1]  = (x == true) and 1 or 0
   in_vec[2]  = (y == true) and 1 or 0

   --nn.build(torch.Tensor({2,1}))
   nn.build({2,1})
   theta = nn.getLayer(1)
   --theta[1] = torch.Tensor({{20,20,-10}}):t()
   theta[1] = 20
   theta[2] = 20
   theta[3] = -10

   return ((nn.forward (in_vec))[1][1] > 0.5)
end

function logicGates.NOT(x)

   --convert boolean to float
   local in_vec = torch.Tensor(1,1)
   in_vec[1]  = (x == true) and 1 or 0

   nn.build({1,1})
   theta = nn.getLayer(1)
   --theta[1] = torch.Tensor({{-20,10}}):t()
   theta[1] = -20
   theta[2] = 10
   return ((nn.forward (in_vec))[1][1] > 0.5)
end

function logicGates.XOR(x,y)
   or_t = logicGates.OR(x,y)
   and_t = logicGates.AND(x,y)
   nand_t = logicGates.NOT(and_t)

   return logicGates.AND(or_t, nand_t)
end

return logicGates

