torch.setdefaulttensortype('torch.FloatTensor')
local nn = require 'nn'

local net = nn.Sequential()

-- compute the size of output layer from input dim/size
local in_dim = 784
local num_class = 10 -- mnist

-- first layer
net:add(nn.LinearPrune(in_dim,100,0.1))
net:add(nn.Sigmoid())

-- second layer
net:add(nn.LinearPrune(100,10,0.1))
net:add(nn.Sigmoid())

-- third layer
--net:add(nn.Linear(10,num_class))
--net:add(nn.Sigmoid())

return net
