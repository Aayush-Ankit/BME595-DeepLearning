--This is the API for digit recognition on MNIST dataset
torch.setdefaulttensortype('torch.FloatTensor')
require 'gnuplot'
--require 'cunn'
local nn = require 'nn'
local img2obj = {}

-- retreive the cifar-100 data
local path = '/home/aa/BME595-DeepLearning/'
local trainset = torch.load(path..'cifar10-train.t7')
local testset = torch.load(path..'cifar10-test.t7')

local trainset_size = (#trainset.data)[1]
--local testset_size = (#testset.data)[1]
local testset_size = 500

-- training spec. variables
local batch_size = 100
local max_iter = 20000 -- one mini-batch per iteration
local eta = 0.015
local epsilon = 0.02

-- user set parameters exclusive for pruning
local if_prune = 1
local num_layers = 6 -- no layers that can be pruned (lenet - 5, lenet_mod = 6)
local prune_iter = 5000

-- other parameters for pruning
local net5 = {1,4,8,10,12}
local net6 = {1,4,6,10,12,14}
local pth5 = torch.Tensor{0,0,0,0,0}
local pth6 = torch.Tensor{0,0,0,0,0,0}
local pth5_inc = 0.01* torch.Tensor{0.00025, 0.00015, 0.00008, 0.00015, 0.00015} -- prune threshold increment
local pth6_inc = 0.05* torch.Tensor{0.00035, 0.00025, 0.0001, 0.0002, 0.00015, 0.0001} -- prune threshold increment
local stats5 = torch.zeros(#net5) -- stores the current network pruning stats
local stats6 = torch.zeros(#net6)

-- network parameters
local in_dim = 3
local in_size = 32
local num_class = 10

-- function to convert a labels to onehot encodings
local function onehot (x)
   local label = torch.zeros(num_class)
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
net = require 'mymodels/plenet_mod'
--net = net:cuda()

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

-- function to train the neural network
function img2obj.train ()

   -- time the training
   print ("If Prune: ", if_prune)
   local time = sys.clock()

   -- local variables to store training outcomes
   local cv_error = torch.zeros(max_iter)
   local tr_error = torch.zeros(max_iter)
   local cv_out = torch.zeros(num_class, testset_size) -- this term must be cuda tensor
   local label_cv_t = torch.zeros(num_class, testset_size)
   local cv_acc = torch.zeros(max_iter)
   local perm
   local stop_iter

   -- define the loss fucntion (error function)
   local loss = nn.MSECriterion()
   --loss = loss:cuda()

   local invec_tt
   local label_tt

   -- begin training
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
         local data = (trainset.data[{{idx}, {}, {}, {}}]):float()
         data = data/255
         invec_t[{{} , {}, {}, i}] = data:view(in_dim, in_size, in_size)
         label_t[{{} ,i}] = onehot(trainset.label[idx])
      end

      -- training with batches
      net:zeroGradParameters() -- make the parameters zero before each batch
      for k = 1,batch_size do
         invec_tt = invec_t[{{} , {}, {}, k}]
         label_tt = label_t[{{} ,k}]

         invec_tt = invec_tt
         label_tt = label_tt

         local tr_out = net:forward(invec_tt)
         tr_error[j] = loss:forward(tr_out, label_tt)
         local gradLoss = loss:backward(tr_out, label_tt)
         net:backward(invec_tt, gradLoss)
      end
      net:updateParameters(eta)

      -- run the model on crossvalidation set to find jcv and accuracy - fix the
      -- cv_error ; acc and average
      for m = 1,testset_size do
         local data_cv = ((testset.data[{{m}, {}, {}, {}}]):float())/255
         data_cv = data_cv:view(in_dim, in_size, in_size) -- not needed
         label_cv_t[{{}, m}] = onehot(testset.label[m])
         local label_cv_tt = label_cv_t[{{}, m}]

         invec_tt = data_cv
         label_tt = label_cv_tt

         cv_out[{{}, m}] = net:forward(invec_tt)
         cv_error[j] = loss:forward(cv_out[{{}, m}], label_tt)
      end

      -- store th cv accuracy
      cv_acc[j] = pred_acc (cv_out, label_cv_t)

      -- stopping condition for training - error/accuracy
      --if ((cv_error[j] < epsilon) or (j == max_iter)) then break end
      if ((cv_acc[j] > 63) or (j == max_iter)) then
         print ("No of iteration to converge CNN:", j)
         time = sys.clock() - time
         print (time)
         stop_iter = j
         break
      end

      -- scan the results at the end of an iteration
      print ('Iteration: ', j, 'cv_acc', cv_acc[j])
   end

    -- plot the accuracy and errors before thye function exits
    gnuplot.figure(1)
    gnuplot.plot('Classification Accurcay', cv_acc[{{1,stop_iter}}])

    gnuplot.figure(2)
    gnuplot.plot({'Training Error',tr_error[{{1,stop_iter}}]}, {'Crossvalidation Error',cv_error[{{1,stop_iter}}]})

end

function img2obj.forward (x)
    --vectorize and normalize the input before sending to the trained NN
    -- maybe the view is unnecessary
    local data = ((x:float())/255)
    -- do a forwrad pass across the trained nn
    local dgt_out = net:forward(data)
    return tostring(pred_digit(dgt_out))
end

function img2obj.view (x)
   -- locally import the library
   local image = require 'image'
   -- predict the output
   local pred = img2obj.forward(x)
   -- diplay the image and the string
   image.display({image = image.drawText(x, pred, 1,1, {color = {255, 0, 0}, size = 2}), zoom = 10})
end

function img2obj.cam (idx)
   -- require the camera
   local camera = require 'camera'
   cam = image.Camera(idx)
   frame = cam:forward()
   cam:stop()

   -- predict the data using the neural network
   data = image.scale(frame, 32, 32)
   img2obj.view(data)
end

return img2obj
