--This  is  the  API  for Neural Network

torch.setdefaulttensortype('torch.FloatTensor')
local NeuralNetwork = {} -- NN library
local tbl = {} -- network parameters
local out = {} -- neruon outputs of each layer for forward propagartion
local de_dtheta = {} -- err_gradients computer by back-prop

--destroys any previosly table instance
local function destroy ()
   tbl = {}
end

function NeuralNetwork.build (x)
   -- generates a random distribution with the specified std. dev.
   destroy()
   for i = 1,(#x-1) do
      tbl[#tbl+1] = (torch.randn(x[i]+1,x[i+1]))/(torch.pow(x[i]+1,0.5))
   end
   return tbl
end

-- return the specified layer
function NeuralNetwork.getLayer (x)
   return tbl[x]
end

function NeuralNetwork.forward (x)
   out[1] = x:t()
   for i = 1,(#tbl) do
      temp = tbl[i]
      out[i+1] = torch.cat(out[i], torch.ones(out[i]:size(1),1),2)* temp
      out[i+1] = (((out[i+1]:mul(-1)):exp()):add(1)):pow(-1)
   end

   return out[#tbl+1]:t()
end

-- HW3 starts here - Sept21
-- function to compute gradient of error in Step 3 depending on error function
local function error_gradient (x)
   -- assuming MSE as error function
   local result = ((out[#tbl+1]):t() - x)
   return result
end

-- function to compute the gradient of sigmoid function
local function sigmoid_gradient (x)
   return (torch.cmul (x, (1-x))):t()
end

function NeuralNetwork.backward (x)
   -- de_dtheta table contains de/dtheta for all thetas
   -- local_error table contains local_erros for each layer
   local local_error = {}

   --compute local error for output layer
   local err_grad = error_gradient(x)
   local sig_grad = sigmoid_gradient(out[#tbl+1])
   local_error[#tbl+1] = torch.cmul(err_grad, sig_grad)
   local_error[#tbl+1] = torch.cat(local_error[#tbl+1], torch.ones(1, out[#tbl+1]:size(1)) ,1)

   -- back-prop to compute local error for layers other than output layer
   -- in every layer, last neuron is the bias term
   for i = (#tbl),1,-1 do
      -- theta' is my table entry itself, so no need to transpose
      local p1 = tbl[i] * (local_error[i+1])[{{1,-2}, {}}]
      local out_temp = torch.cat(out[i], torch.ones(out[i]:size(1),1), 2)
      local p2 = (sigmoid_gradient(out_temp))
      local_error[i] = torch.cmul (p1,p2)
   end

   for i = 1,(#tbl) do
      local local_err_temp = (local_error[i+1])[{{1,-2} , {}}]
      local a_hat = torch.cat(out[i], torch.ones(out[i]:size(1),1),2)
      de_dtheta[i] = (local_err_temp * a_hat) / (out[i]:size(2))
      --de_dtheta[i] = (local_err_temp * a_hat)
   end
end

-- functions to update parameters
function NeuralNetwork.updateParams (eta)
   for i =1,(#tbl) do
      --print (tbl[i])
      tbl[i] = (tbl[i]:t() - (eta*de_dtheta[i])):t()
      de_dtheta[i]:zero()
      --print (tbl[i])
   end
end

return NeuralNetwork
