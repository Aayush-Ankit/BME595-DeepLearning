--This is the API for logic gates
torch.setdefaulttensortype('torch.FloatTensor')
--require 'gnuplot'
local nn = require 'NeuralNetwork'

--local logicGate tables
local lg = {}
local AND = {} local OR = {} local NOT = {} local XOR = {}
lg['AND'] = AND lg['OR'] = OR lg['NOT'] = NOT lg['XOR'] = XOR

-- local variables
local theta -- upon the function call getLayer this refrences the theta table

-- other local variable
local eta = 0.03 -- learning rate
local max_iter = 1000000
local epsilon = 0.08
local lst_idx

-- All the training assumes SGD - AND, OR, NOT, XOR
-- AND Table Methods/functions
function AND.train ()
   -- generating the AND labels
   local in_vec = torch.randn(2,4)
   in_vec[1][1] = 1 in_vec[2][1] = 1
   in_vec[1][2] = 1 in_vec[2][2] = 0
   in_vec[1][3] = 0 in_vec[2][3] = 1
   in_vec[1][4] = 0 in_vec[2][4] = 0
   local label  = torch.randn(1,4)
   label[1][1] = 1 label[1][2] = 0 label[1][3] = 0 label[1][4] = 0

   --store all the models, corrsponding cv_error and tr_error
   local model = {}
   local min_cv_idx
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)

   --build the neural network
   nn.build({2,1})

   -- start training
   for j = 1,max_iter do
      --generate the input vec and label for training on the fly
      local perm = torch.randperm(4)
      local in_vec_t = torch.randn(2,4)
      local label_t = torch.randn(1,4)
      for i = 1,4 do
         in_vec_t[{{}, i}] = in_vec[{{}, perm[i]}]
         label_t[{{}, i}] = label[{{}, perm[i]}]
      end

      --forward, backward and updateparam
      local tr_out = nn.forward (in_vec_t)
      nn.backward (label_t)
      nn.updateParams (eta)

      -- storing the trained model and training error
      model[j] = nn.getLayer(1)
      tr_error[j] = (((tr_out-label_t):pow(2)):sum())/(tr_out:size(2))

      -- store the cross-validation error and track the min cv_error
      local cv_out = nn.forward(in_vec)
      cv_error[j] = (((cv_out-label):pow(2)):sum())/(cv_out:size(2))
      if (j == 1) then
         min_cv_idx = j
      elseif (cv_error[j] < cv_error[min_cv_idx]) then
         min_cv_idx = j
      end

      -- view the errors after each iteration
      --print ("Iteration:", j)
      --print ("tr_error:", tr_error[j])
      --print ("cv_error:", cv_error[j])
      --print ("min_cv_idx", min_cv_idx)

      -- stopping criterion for training (output the model with lowest cv_error)
      if (tr_error[j] < epsilon) then
         lst_idx = j
         break
      end
   end

   -- assign the best model to theta
   theta = nn.getLayer(1)
   theta[1] = model[min_cv_idx][1]
   theta[2] = model[min_cv_idx][2]
   theta[3] = model[min_cv_idx][3]
   --print ("Last Iteration:",lst_idx)
   --gnuplot.figure(1)
   --gnuplot.plot({'AND Training Error',tr_error[{{1,lst_idx}}]}, {'AND Crossvalidation Error',cv_error[{{1,lst_idx}}]})
end

function AND.set () -- Sets the user defined values as tehta for reproducing HW02 results
   --build a networki
   nn.build({2,1})
   --set the thetas, thetas get passed to the nn theta by reference
   theta = nn.getLayer(1)
   theta[1] = 20
   theta[2] = 20
   theta[3] = -30
end

function AND.forward (x, y) -- Propagates an input acrss a network, done after set has been used
   local in_vec = torch.Tensor(2,1)
   in_vec[1]  = (x == true) and 1 or 0
   in_vec[2]  = (y == true) and 1 or 0
   return (((nn.forward (in_vec))[1][1]) > 0.5)
end


-- OR Table Methods/functions
function OR.train ()
   -- generating the AND labels
   local in_vec = torch.randn(2,4)
   in_vec[1][1] = 1 in_vec[2][1] = 1
   in_vec[1][2] = 1 in_vec[2][2] = 0
   in_vec[1][3] = 0 in_vec[2][3] = 1
   in_vec[1][4] = 0 in_vec[2][4] = 0
   local label  = torch.randn(1,4)
   label[1][1] = 1 label[1][2] = 1 label[1][3] = 1 label[1][4] = 0

   --store all the models, corrsponding cv_error and tr_error
   local model = {}
   local min_cv_idx
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)

   --build the neural network
   nn.build({2,1})

   -- start training
   for j = 1,max_iter do
      --generate the input vec and label for training on the fly
      local perm = torch.randperm(4)
      local in_vec_t = torch.randn(2,4)
      local label_t = torch.randn(1,4)
      for i = 1,4 do
         in_vec_t[{{}, i}] = in_vec[{{}, perm[i]}]
         label_t[{{}, i}] = label[{{}, perm[i]}]
      end

      --forward, backward and updateparam
      local tr_out = nn.forward (in_vec_t)
      nn.backward (label_t)
      nn.updateParams (eta)

      -- storing the trained model and training error
      model[j] = nn.getLayer(1)
      tr_error[j] = (((tr_out-label_t):pow(2)):sum())/(tr_out:size(2))

      -- store the cross-validation error and track the min cv_error
      local cv_out = nn.forward(in_vec)
      cv_error[j] = (((cv_out-label):pow(2)):sum())/(cv_out:size(2))
      if (j == 1) then
         min_cv_idx = j
      elseif (cv_error[j] < cv_error[min_cv_idx]) then
         min_cv_idx = j
      end

      -- view the errors after each iteration
      --print ("Iteration:", j)
      --print ("tr_error:", tr_error[j])
      --print ("cv_error:", cv_error[j])
      --print ("min_cv_idx", min_cv_idx)

      -- stopping criterion for training (output the model with lowest cv_error)
      if (tr_error[j] < epsilon) then
         lst_idx = j
         break
      end
   end

   -- assign the best model to theta
   theta = nn.getLayer(1)
   theta[1] = model[min_cv_idx][1]
   theta[2] = model[min_cv_idx][2]
   theta[3] = model[min_cv_idx][3]
   --print ("Last Iteration:",lst_idx)
   --gnuplot.figure(2)
   --gnuplot.plot({'OR Training Error',tr_error[{{1,lst_idx}}]}, {'OR Crossvalidation Error',cv_error[{{1,lst_idx}}]})
end

function OR.set () -- Sets the user defined values as tehta for reproducing HW02 results
   --build a networki
   nn.build({2,1})
   --set the thetas, thetas get passed to the nn theta by reference
   theta = nn.getLayer(1)
   theta[1] = 20
   theta[2] = 20
   theta[3] = -10
end

function OR.forward (x, y) -- Propagates an input acrss a network, done after set has been used
   local in_vec = torch.Tensor(2,1)
   in_vec[1]  = (x == true) and 1 or 0
   in_vec[2]  = (y == true) and 1 or 0
   return (((nn.forward (in_vec))[1][1]) > 0.5)
end


-- NOT Table Methods/functions
function NOT.train ()
   -- generating the NOT dataset labels
   local in_vec = torch.randn(1,2)
   in_vec[1][1] = 1 in_vec[1][2] = 0
   local label  = torch.randn(1,2)
   label[1][1] = 0 label[1][2] = 1

   --store all the models, corrsponding cv_error and tr_error
   local model = {}
   local min_cv_idx
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)

   --build the neural network
   nn.build({1,1})

   -- start training
   for j = 1,max_iter do
      --generate the input vec and label for training on the fly
      local perm = torch.randperm(2)
      local in_vec_t = torch.randn(1,2)
      local label_t = torch.randn(1,2)
      for i = 1,2 do
         in_vec_t[{{}, i}] = in_vec[{{}, perm[i]}]
         label_t[{{}, i}] = label[{{}, perm[i]}]
      end

      --forward, backward and updateparam
      local tr_out = nn.forward (in_vec_t)
      nn.backward (label_t)
      nn.updateParams (eta)

      -- storing the trained model and training error
      model[j] = nn.getLayer(1)
      tr_error[j] = (((tr_out-label_t):pow(2)):sum())/(tr_out:size(2))

      -- store the cross-validation error and track the min cv_error
      local cv_out = nn.forward(in_vec)
      cv_error[j] = (((cv_out-label):pow(2)):sum())/(cv_out:size(2))
      if (j == 1) then
         min_cv_idx = j
      elseif (cv_error[j] < cv_error[min_cv_idx]) then
         min_cv_idx = j
      end

      -- view the errors after each iteration
      --print ("Iteration:", j)
      --print ("tr_error:", tr_error[j])
      --print ("cv_error:", cv_error[j])
      --print ("min_cv_idx", min_cv_idx)

      -- stopping criterion for training (output the model with lowest cv_error)
      if (tr_error[j] < epsilon) then
         lst_idx = j
         break
      end
   end

   -- assign the best model to theta
   theta = nn.getLayer(1)
   theta[1] = model[min_cv_idx][1]
   theta[2] = model[min_cv_idx][2]
   --print ("Last Iteration:",lst_idx)
   --gnuplot.figure(3)
   --gnuplot.plot({'NOT Training Error',tr_error[{{1,lst_idx}}]}, {'NOT Crossvalidation Error',cv_error[{{1,lst_idx}}]})
end

function NOT.set () -- Sets the user defined values as tehta for reproducing HW02 results
   --build a network
   nn.build({1,1})
   --set the thetas, *thetas get passed to the nn theta by reference
   theta = nn.getLayer(1)
   theta[1] = -20
   theta[2] = 10
end

function NOT.forward (x) -- Propagates an input acrss a network, done after set has been used
   local in_vec = torch.Tensor(1,1)
   in_vec[1]  = (x == true) and 1 or 0
   return (((nn.forward (in_vec))[1][1]) > 0.5)
end


-- XOR Table Methods/functions
function XOR.train ()
   -- generating the XOR dataset and labels
   local in_vec = torch.randn(2,4)
   in_vec[1][1] = 1 in_vec[2][1] = 1
   in_vec[1][2] = 1 in_vec[2][2] = 0
   in_vec[1][3] = 0 in_vec[2][3] = 1
   in_vec[1][4] = 0 in_vec[2][4] = 0
   local label  = torch.randn(1,4)
   label[1][1] = 0 label[1][2] = 1 label[1][3] = 1 label[1][4] = 0

   --store all the models, corrsponding cv_error and tr_error
   local model = {}
   local min_cv_idx
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)

   --build the neural network
   nn.build({2,2,1})

   -- start training
   for j = 1,max_iter do
      --generate the input vec and label for training on the fly
      local perm = torch.randperm(4)
      local in_vec_t = torch.randn(2,4)
      local label_t = torch.randn(1,4)
      for i = 1,4 do
         in_vec_t[{{}, i}] = in_vec[{{}, perm[i]}]
         label_t[{{}, i}] = label[{{}, perm[i]}]
      end

      --forward, backward and updateparam
      local tr_out = nn.forward (in_vec_t)
      nn.backward (label_t)
      nn.updateParams (eta)

      -- storing the trained model and training error
      model[j] = nn.getLayer(1)
      tr_error[j] = (((tr_out-label_t):pow(2)):sum())/(tr_out:size(2))

      -- store the cross-validation error and track the min cv_error
      local cv_out = nn.forward(in_vec)
      cv_error[j] = (((cv_out-label):pow(2)):sum())/(cv_out:size(2))
      if (j == 1) then
         min_cv_idx = j
      elseif (cv_error[j] < cv_error[min_cv_idx]) then
         min_cv_idx = j
      end

      -- view the errors after each iteration
      --print ("Iteration:", j)
      --print ("tr_error:", tr_error[j])
      --print ("cv_error:", cv_error[j])
      --print ("min_cv_idx", min_cv_idx)

      -- stopping criterion for training (output the model with lowest cv_error)
      if (tr_error[j] < epsilon) then
         lst_idx = j
         break
      end
   end

   -- assign the best model to theta
   theta = nn.getLayer(1)
   theta[1] = model[min_cv_idx][1]
   theta[2] = model[min_cv_idx][2]
   theta[3] = model[min_cv_idx][3]
   --print ("Last Iteration:",lst_idx)
   --gnuplot.figure(4)
   --gnuplot.plot({'XOR Training Error',tr_error[{{1,lst_idx}}]}, {'XOR Crossvalidation Error',cv_error[{{1,lst_idx}}]})
end

function XOR.set () -- Sets the user defined values as tehta for reproducing HW02 results
   --build a networki
   nn.build({2,2,1})
   --set the thetas, thetas get passed to the nn theta by reference
   local theta1 = nn.getLayer(1)
   theta1[1][1] = 20  theta1[2][1] = 20  theta1[3][1] = -10
   theta1[1][2] = -20 theta1[2][2] = -20 theta1[3][2] = 30
   local theta2 = nn.getLayer(2)
   theta2[1] = 20 theta2[2] = 20 theta2[3] = -30 --AND

end

function XOR.forward (x, y) -- Propagates an input acrss a network, done after set has been used
   local in_vec = torch.Tensor(2,1)
   in_vec[1]  = (x == true) and 1 or 0
   in_vec[2]  = (y == true) and 1 or 0
   return (((nn.forward (in_vec))[1][1]) > 0.5)
end

return lg
