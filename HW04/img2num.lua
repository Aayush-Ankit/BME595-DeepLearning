--This is the API for digit recognition on MNIST dataset
torch.setdefaulttensortype('torch.FloatTensor')
require 'gnuplot'
local nn = require 'nn'
local img2num = {}

-- retreive the mnist data
local mnist = require 'mnist'
local trainset = mnist.traindataset()
local testset = mnist.testdataset()
local trainset_size = trainset.size
--local testset_size = testset.size
local testset_size = 500

-- training spec. variables
local batch_size = 60
local max_iter = 1000000 -- one mini-batch per iteration
local eta = 0.15
local epsilon = 0.05

-- network parameters
local in_size = 784
local num_hidden = 50
local num_class = 10
local net_size = {in_size, num_hidden, num_hidden, num_class}
local num_layers = #(net_size)-1

-- function to convert a labels to onehot encodings
local function onehot (x)
   local label = torch.zeros(10)
   label[x+1] = 1
   return label
end

-- function to predict the output digit (ineteger) from the neuron outputs
local function pred_digit (x)
   local prob, pred
   prob, pred = torch.max(x,1)
   return (pred-1)
end

-- function to evaluate the classfication accuracy
local function pred_acc (x, labels)
   local pred = pred_digit(x)
   local labels = pred_digit(labels)
   return (torch.eq(pred:byte(), labels:byte()):sum())/x:size(2)*100
end

-- build the neural network
local net = nn.Sequential()
for i = 1,(#(net_size)-1) do
   net:add(nn.Linear(net_size[i], net_size[i+1]))
   net:add(nn.Sigmoid())
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

   -- define the loss fucntion (error function)
   local loss = nn.MSECriterion()

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

      -- training with batches
      net:zeroGradParameters() -- make the parameters zero before each batch
      for k = 1,batch_size do
         local invec_tt = invec_t[{{} ,k}]
         local label_tt = label_t[{{} ,k}]
         local tr_out = net:forward(invec_tt)
         tr_error[j] = loss:forward(tr_out, label_tt)
         local gradLoss = loss:backward(tr_out, label_tt)
         net:backward(invec_tt, gradLoss)
      end
      net:updateParameters(eta)

      -- run the model on crossvalidation set to find jcv and accuracy
      local label_cv_t = torch.zeros(num_class, testset_size)
      local cv_out = torch.zeros(num_class, testset_size)
      for i = 1,testset_size do
         local data = (((testset[i].x):float())/255):view(in_size)
         label_cv_t[{{}, i}] = onehot(testset[i].y)
         cv_out[{{}, i}] = net:forward(data)
         cv_error[j] = loss:forward(cv_out[{{}, i}], label_cv_t[{{}, i}])
      end

      -- stopping condition for training
      --if (cv_error[j] < epsilon) then break end

      -- store th cv accuracy
      cv_acc[j] = pred_acc (cv_out, label_cv_t)
      gnuplot.figure(1)
      gnuplot.plot('Classification Accurcay', cv_acc[{{1,j}}])

      --print ("Iteration:", k)
      --print ("tr_error:", tr_error[k-1])
      gnuplot.figure(2)
      gnuplot.plot({'Training Error',tr_error[{{1,j}}]}, {'Crossvalidation Error',cv_error[{{1,j}}]})
   end

end

function img2num.forward (x)
         local data = (x:float()):view(in_size)
         data = data/255

         -- do a forwrad pass across the trained nn
         local dgt_out = net:forward(data)
         return pred_digit(dgt_out)
end

return img2num
