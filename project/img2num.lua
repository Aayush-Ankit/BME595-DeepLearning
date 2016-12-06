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
local batch_size = 10
local max_iter = 10000 -- one mini-batch per iteration
local eta = 0.10

-- user set parameters exclusive for pruning
local if_prune = 1
local num_layers = 5 -- no layers that can be pruned (lenet - 5, lenet_mod = 6)
local prune_iter = 3000

-- other parameters for pruning
local net5 = {1,4,8,10,12}
local net6 = {1,4,6,10,12,14}
local pth5 = torch.Tensor{0,0,0,0,0}
local pth6 = torch.Tensor{0,0,0,0,0,0}
local pth5_inc = 0.01* torch.Tensor{0.00025, 0.00015, 0.00008, 0.00015, 0.00015} -- prune threshold increment
local pth6_inc = torch.Tensor{0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001} -- prune threshold increment
local stats5 = torch.zeros(#net5) -- stores the current network pruning stats
local stats6 = torch.zeros(#net6)

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
local net = require 'mymodels/plenet'

-- function to prune the network
local function do_prune ()

  if (num_layers == 5) then
    pth5 = pth5 + pth5_inc
    print ("prune th: ", pth5)
    for layer = 1, #(net5) do
      local ltemp = net:get(net5[layer])
      ltemp:updatepth(pth5[layer])
      stats5[layer] = ltemp:SeeMap()
    end
    print ("Prune Statistics: ", stats5)
  end

  if (num_layers == 6) then
    pth6 = pth6 + pth6_inc
    print ("prune th: ", pth6)
    for layer = 1, #(net6) do
      local ltemp = net:get(net6[layer])
      ltemp:updatepth(pth6[layer])
      stats6[layer] = ltemp:SeeMap()
    end
    print ("Prune Statistics: ", stats6)
  end
  print ("Prune Success!")
end

-- network parameters
local in_dim = 1
local in_size = 28
local num_class = 10

-- function to train the neural network
function img2num.train ()

   -- local variables to store training outcomes
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)
   local cv_out = torch.zeros(num_class, testset_size)
   local label_cv_t = torch.zeros(num_class, testset_size)
   local cv_acc = torch.zeros(max_iter)
   local perm
   local stop_iter

   -- define the loss fucntion (error function)
   local loss = nn.MSECriterion()

   -- begin training
   print ("Training")
   print ("If Prune: ", if_prune)
   local time = sys.clock()

   for j = 1,max_iter do

      -- prune if flag is set
      if ((j>prune_iter) and (if_prune==1)) then do_prune() end

      -- change the permutation after the entire dataset has been traversed
      if ((((j-1) * batch_size) % trainset_size) == 0) then
         perm = torch.randperm(trainset_size)

      end

      -- generating the required form of input and output
      local invec_t = torch.zeros(in_dim, in_size, in_size, batch_size)
      local label_t = torch.zeros(num_class, batch_size)
      for i = 1,batch_size do
         local idx = perm[(((j-1) * batch_size) % trainset_size) + i]
         local data = (trainset[idx].x):float()
         data = data/255
         --invec_t[{{} ,i}] = data:view(in_size,1)
         invec_t[{{} , {}, {}, i}] = data:view(in_dim, in_size, in_size)
         label_t[{{} ,i}] = onehot(trainset[idx].y)
      end

      -- training with batches
      net:zeroGradParameters() -- make the parameters zero before each batch
      for k = 1,batch_size do
         local invec_tt = invec_t[{{} , {}, {}, k}]
         local label_tt = label_t[{{} ,k}]

         local tr_out = net:forward(invec_tt)
         tr_error[j] = loss:forward(tr_out, label_tt)
         local gradLoss = loss:backward(tr_out, label_tt)
         net:backward(invec_tt, gradLoss)
      end
      net:updateParameters(eta)

      -- run the model on crossvalidation set to find jcv and accuracy
      for i = 1,testset_size do
         local data = (((testset[i].x):float())/255):view(in_dim, in_size, in_size)
         label_cv_t[{{}, i}] = onehot(testset[i].y)
         cv_out[{{}, i}] = net:forward(data)
         cv_error[j] = loss:forward(cv_out[{{}, i}], label_cv_t[{{}, i}])
      end

      -- store th cv accuracy
      cv_acc[j] = pred_acc (cv_out, label_cv_t)

      -- stopping condition for training
      if ((cv_acc[j] > 99) or (j == max_iter)) then
         print ("No of iteration to converge CNN:", j)
         stop_iter = j
         time = sys.clock() - time
         print (time)
         break
      end


      print ("Iteration:", j, "cv_acc:", cv_acc[j])
      --print ("cv_error:", cv_error[j])
      --print ("cv_acc:", cv_acc[j])
   end

    -- plot the accuracy and errors
    gnuplot.figure(1)
    gnuplot.plot('Classification Accurcay', cv_acc[{{1,stop_iter}}])
    gnuplot.pngfigure('mnist_cnn_acc')

    gnuplot.figure(2)
    gnuplot.plot({'Training Error',tr_error[{{1,stop_iter}}]}, {'Crossvalidation Error',cv_error[{{1,stop_iter}}]})

end

function img2num.forward (x)

         --vectorize and normalize the input before sending to the trained NN
         local data = ((x:float())/255):view(in_dim, in_size, in_size)

         -- do a forwrad pass across the trained nn
         local dgt_out = net:forward(data)
         return pred_digit(dgt_out)
end

return img2num
