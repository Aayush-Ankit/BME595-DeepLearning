--This  is  the  API  for Neural Network

torch.setdefaulttensortype('torch.FloatTensor')
local NeuralNetwork = {}
--tbl = {}
-- built the network
local function destroy ()
   tbl = {}
end

function NeuralNetwork.build (x)
   -- create an empty table
   --local tbl = {}
   destroy()
   for i = 1,(x:size(1)-1) do
      --random tensor with zero mean and specified std. dev.
      tbl[#tbl+1] = torch.randn(x[i]+1,x[i+1])
   end
   return tbl
end

-- return the specified layer
function NeuralNetwork.getLayer (x)
   return tbl[x]
end

function NeuralNetwork.forward (x)
   local out = x:t()
   for i = 1,(#tbl) do
      temp = tbl[i]
      out = torch.cat(out, torch.ones(out:size(1),1),2)* temp
      out = (((out:mul(-1)):exp()):add(1)):pow(-1)
   end

   return out
end

return NeuralNetwork







