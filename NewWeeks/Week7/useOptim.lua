-- This script uses optimization packages to train a XOR gate
require 'nn'
require 'optim'
require 'cunn'
require 'xlua'

-- create the model
torch.manualSeed(1234) -- get the same random initialization evergy trial
local model = nn.Sequential()
local n = 2
local K = 1
local s = {n,1000,1000,K}
model:add(nn.Linear(s[1], s[2]))
model:add(nn.Tanh())
model:add(nn.Linear(s[2], s[3]))
model:add(nn.Tanh())
model:add(nn.Linear(s[3], s[4]))
--model:add(nn.Tanh())
--model:add(nn.Linear(s[2], s[3]))

local loss = nn.MSECriterion() -- set the type of loss function

-- create input data - Put the training data in GPU memory
local m = 128
local X = torch.DoubleTensor(m,n)
local Y = torch.DoubleTensor(m)

for i = 1, m do
  local x = torch.randn(2)
  local y = x[1]*x[2] > 0 and -1 or 1
  X[i]:copy(x) -- fine also for cuda
  Y[i] = y
end

-- GPU - send the network and data to device(gpu) memory
model:cuda()
loss:cuda()
X = X:cuda()
Y = Y:cuda()

local theta, gradTheta = model:getParameters()

local optimState = {learningRate = 0.015}

local time0 = sys.clock()
for epoch = 1, 1e3 do
  -- show the pregressbar
  --xlua.progress (epoch, 1e3)

  -- training function
  function feval(theta)
    gradTheta:zero() -- same as model:zeroGradParameters()
    local h_x = model:forward(X)
    local J = loss:forward(h_x, Y)
    print (J) -- for debugging - see the errors
    local dJ_dh_x = loss:backward(h_x, Y)
    model:backward(X, dJ_dh_x)
    return J, gradTheta
  end
  optim.sgd(feval, theta, optimState)
end
time = sys.clock() - time0
print ("25.87")
print ("Execution Time: ", time)

net = model
