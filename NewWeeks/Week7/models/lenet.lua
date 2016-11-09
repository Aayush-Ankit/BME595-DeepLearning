torch.setdefaulttensortype('torch.FloatTensor')
local nn = require 'nn'

local net = nn.Sequential()

-- compute the size of output layer from input dim/size
local in_size = 32
local num_class = 10 -- mnist
local in_dim = 3
local inp = torch.Tensor(in_dim, in_size, in_size)

-- first layer
net:add(nn.SpatialConvolution(in_dim,6,5,5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Tanh())

-- second layer
net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Tanh())

-- to find the size of fcn layer
local out_size = #(net:forward(inp))

-- reshape into a linear tensor
net:add(nn.View(-1))

-- output layer
net:add (nn.Linear(out_size[1]*out_size[2]*out_size[3], 120))
net:add(nn.Tanh())
net:add (nn.Linear(120, 84))
net:add(nn.Tanh())


-- final layer
net:add(nn.Linear(84, num_class))

return net
