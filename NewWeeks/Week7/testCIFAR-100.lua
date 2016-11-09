--Test Script for LeNet on CIFAR
require 'torch'
require 'sys'
torch.setdefaulttensortype('torch.FloatTensor')
--img2num = require 'img2num'
img2obj = require 'img2obj'
-- train the nn
--img2num.train()
img2obj.train()

-- test the trained nn using forward function
--[[local path = '/home/aa/BME595-DeepLearning/'
local testset = torch.load(path..'cifar100-test.t7')
local testset_size = 10

local pred = {}
local actual = torch.zeros(1,testset_size)
local in_dim = (#testset.data)[2]
local in_size = (#testset.data)[3]
local data = torch.zeros(in_dim, in_size, in_size)

for i = 1, testset_size do
   data = (testset.data[{{i}, {}, {}, {}}]):view(in_dim, in_size, in_size)
   pred[#pred +1] = img2obj.forward(data)
   actual[1][i] = testset.label[i]
end

print ("Actual :", actual)
print ("Prediction :",pred)

-- test the view function
data = (testset.data[{{1}, {}, {}, {}}]):view(in_dim, in_size, in_size)
img2obj.view (data)

-- test the cam function
local my_idx = 0
img2obj.cam(my_idx)--]]
