--Test Script for logic gates

torch.setdefaulttensortype('torch.FloatTensor')
--img2num = require 'img2num'
img2num = require 'img2num_mod'

-- train the nn
img2num.train()

-- test the trained nn using forward function
local mnist = require 'mnist'
local testset = mnist.testdataset()
--local testset_size = testset.size
local testset_size = 10

-- Test the forward function
local pred = torch.zeros(1,testset_size)
local actual = torch.zeros(1,testset_size)

for i = 1, testset_size do
   local input = testset[i].x
   pred[1][i] = img2num.forward(input)
   actual[1][i] = testset[i].y
end

print ("Actual :", actual)
print ("Prediction :", pred)


