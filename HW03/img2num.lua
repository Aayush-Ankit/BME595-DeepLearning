--This is the API for digit recognition on MNIST dataset
torch.setdefaulttensortype('torch.FloatTensor')
--require 'gnuplot'
local nn = require 'NeuralNetwork'
local img2num = {}

-- retreive the mnist data
local mnist = require 'mnist'
local trainset = mnist.traindataset()
local testset = mnist.testdataset()
local trainset_size = trainset.size
local testset_size = testset.size

-- training spec. variables
local batch_size = 600
local max_iter = 3000 -- one mini-batch per iteration
local eta = 0.3
local epsilon = 0.1

-- network parameters
local in_size = 784
local num_hidden = 10
local num_class = 10
local net_size = {in_size, num_hidden, num_hidden, num_class}
local num_layers = #(net_size)-1
local theta_best = {} -- stores the model with least Jcv

-- function to convert a labels to onehot encodings
local function onehot (x)
   local label = torch.zeros(10)
   label[x+1] = 1
   return label
end

local function pred_digit (x)
   local prob, pred
   prob, pred = torch.max(x,1)
   return (pred-1)
end

local function pred_acc (x, labels)
   local pred = pred_digit(x)
   local labels = pred_digit(labels)
   return (torch.eq(pred:byte(), labels:byte()):sum())/x:size(2)*100
end

-- function to train the neural network
function img2num.train ()

   -- local variables to store training outcomes
   local model = {}
   local min_cv_idx = 1
   local last_iter = 0 -- last iteration before training stopped
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)
   local cv_acc = torch.zeros(max_iter)
   local perm

   -- build the network before training starts
   nn.build (net_size)

   -- begin training
   for j = 1,max_iter do

      -- change the permutation after the entire dataset has been traversed
      if ((((j-1) * batch_size) % trainset_size) == 0) then
         perm = torch.randperm(trainset_size)
      end

      -- generating the required form of input and output
      local invec_t = torch.zeros(in_size, batch_size)
      local label_t = torch.zeros(num_class, batch_size)
      for i = 1,batch_size do
         local idx = perm[(((j-1) * batch_size) % trainset_size) + i]
         local data = (trainset[idx].x):float()
         data = data/255
         --invec_t[{{} ,i}] = data:view(1, -1):t()
         invec_t[{{} ,i}] = data:view(in_size,1)
         label_t[{{} ,i}] = onehot(trainset[idx].y)
      end

      -- training
      local tr_out = nn.forward(invec_t)
      nn.backward (label_t)
      nn.updateParams (eta)
      tr_error[j] = (((tr_out-label_t):pow(2)):sum())/(tr_out:size(2))

      -- run the model on crossvalidation set to find accuracy
      local invec_cv_t = torch.zeros(in_size, testset_size)
      local label_cv_t = torch.zeros(num_class, testset_size)
      for i = 1,testset_size do
         local data = (testset[i].x):float()
         data = data/255
         invec_cv_t[{{} ,i}] = data:view(in_size,1)
         label_cv_t[{{}, i}] = onehot(testset[i].y)
      end

      local cv_out = nn.forward(invec_cv_t)
      cv_error[j] = (((cv_out - label_cv_t):pow(2)):sum())/(cv_out:size(2))

      if (j == 1) then
         min_cv_idx = j
      elseif (cv_error[j] < cv_error[min_cv_idx]) then
         min_cv_idx = j
         -- save the best theta model
         local theta_tbl = {}
         for i = 1,num_layers do
            theta_tbl[i] = nn.getLayer(i)
         end
         theta_best = theta_tbl
      end

      -- store th cv accuracy
      cv_acc[j] = pred_acc (cv_out, label_cv_t)
      --gnuplot.figure(1)
      --gnuplot.plot('Classification Accurcay', cv_acc[{{1,j}}])

      --print ("Iteration:", k)
      --print ("tr_error:", tr_error[k-1])
      --gnuplot.figure(2)
      --gnuplot.plot({'Training Error',tr_error[{{1,j}}]}, {'Crossvalidation Error',cv_error[{{1,j}}]})
   end

end

function img2num.forward (x)
         local data = x:float()
         data = data/255

         -- assign the theta_best to the neural network
         nn.build(net_size)
         for i = 1,num_layers do
            -- get the layer
            theta = nn.getLayer(i)
            -- assign the theta_best to this theta by referencing
            for m = 1,theta:size(1) do
               for n = 1,theta:size(2) do
                  theta[m][n] = theta_best[i][m][n]
               end
            end
         end

         -- do a forwrad pass across the nn
         local dgt_out = nn.forward()
         return pred_digit(dgt_out)
end

return img2num
