--This is the API for digit recognition on MNIST dataset
torch.setdefaulttensortype('torch.FloatTensor')
require 'gnuplot'
local nn = require 'NeuralNetwork'
local img2num = {}

-- retreive the mnist data
local mnist = require 'mnist'
local trainset = mnist.traindataset()
local testset = mnist.testdataset()
local trainset_size = trainset.size
-- local testset_size = testset.size
local testset_size = 1000

-- training spec. variables
local batch_size = 600
local max_iter = 100 -- one mini-batch per iteration
local eta = 0.15
local epsilon = 0.1

-- network parameters
local num_layers = 3
local in_size = 784
local num_hidden = 1000
local num_class = 10
local net_size = {in_size, num_hidden, num_class}

-- function to convert a labels to onehot encodings
local function onehot (x)
   local label = torch.zeros(10)
   label[x+1] = 1
   return label
end

local function pred_digit (x)
   --local  out = torch.linspace(0,9,10)
   --out = out:repeatTensor(testset_size,1):t()
   --out = out:cmul(x)
   local prob, pred
   prob, pred = torch.max(x,1)
   return (pred-1)
end


-- function to train the neural network
function img2num.train ()

   -- local variables to store training outcomes
   local model = {}
   local min_cv_idx = 0
   local last_iter = 0 -- last iteration before training stopped
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)
   local perm = torch.randperm(trainset_size)

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
         local data = (trainset[perm[(((j-1) * batch_size) % trainset_size) + i]].x)
         data = (data:type('torch.FloatTensor'))/255
         invec_t[{{} ,i}] = data:view(in_size,1)
         label_t[{{} ,i}] = onehot(trainset[perm[(((j-1) * batch_size) % trainset_size) + i]].y)
      end

      -- build the neural network
      nn.build (net_size)

      -- training
      local tr_out = nn.forward(invec_t)
      nn.backward (label_t)
      nn.updateParams (eta)

      -- store the current model and results
      local theta_tbl = {}
      for i = 1,num_layers do
         theta_tbl[i] = nn.getLayer(i)
      end
      model[j] = theta_tbl
      --print (tr_out)
      --print (label_t)
      tr_error[j] = (((tr_out-label_t):pow(2)):sum())/(tr_out:size(1) * tr_out:size(2))

      -- view the errors after each iteration
      print ("Iteration:", j, "tr_error:", tr_error[j])
      --gnuplot.plot({'Training Error',tr_error[{{1,j}}]})

      -- run the model on crossvalidation set to find accuracy
      local invec_cv_t = torch.zeros(in_size, testset_size)
      local label_cv_t = torch.zeros(num_class, testset_size)
      for i = 1,testset_size do
         local data = testset[i].x
         data = (data:type('torch.FloatTensor'))/255
         invec_cv_t[{{} ,i}] = data:view(in_size,1)
         label_cv_t[{{}, i}] = onehot(testset[i].y)
      end
      local cv_out = nn.forward(invec_cv_t)
      --local digit_pred = pred_digit(cv_out)
      --digit_pred = digit_pred:type('torch.FloatTensor')
      --local pred_acc = torch.eq (label_cv_t, digit_pred)
      --pred_acc = torch.sum(pred_acc)*100/testset_size
      --print ("CV Accuracy:", pred_acc)
      cv_error[j] = (((cv_out - label_cv_t):pow(2)):sum())/(cv_out:size(1) * cv_out:size(2))
      if (j == 1) then
         min_cv_idx = j
      elseif (cv_error[j] < cv_error[min_cv_idx]) then
         min_cv_idx = j
      end

      gnuplot.plot({'Training Error',tr_error[{{1,j}}]}, {'Crossvalidation Error',cv_error[{{1,j}}]})
   end

end

return img2num
