--torch.setdefaulttensortype('torch.FloatTensor')
-- This is the model for alexnet
nn = require 'nn'

-- make the first branch
local fb1 = nn.Sequential()

-- convolution1
fb1:add (nn.SpatialConvolution (3, 12, 3, 3, 1, 1))
fb1:add (nn.ReLU(true)) -- true flag for inline storage
fb1:add (nn.SpatialMaxPooling (2, 2, 2, 2))

-- convolution2
fb1:add (nn.SpatialConvolution (12, 32, 3, 3, 1,1))
fb1:add (nn.ReLU(true)) -- true flag for inline storage
fb1:add (nn.SpatialMaxPooling (2, 2, 2, 2))

-- convolution3
fb1:add (nn.SpatialConvolution (32, 32, 3, 3, 1, 1))
fb1:add (nn.ReLU(true)) -- true flag for inline storage
fb1:add (nn.SpatialMaxPooling (2, 2, 2, 2))

-- make the second branch - clone the first one
local fb2 = fb1:clone()
--have a different initilaization of conv kernels that fb1 to break symmetry
for k,v in ipairs (fb2:findModules('nn.SpatialConvolution')) do
  v:reset()
end

-- concatenate the two branches along the row dimension
local features = nn.Concat(2)
features:add (fb1)
features:add (fb2)

-- find the size of features
local inp = torch.Tensor(3,32,32)
local size = #(features:forward(inp))

-- define the classifier - fully connected layers
local nClasses = 10
local classifier = nn.Sequential()

classifier:add (nn.View(size[1]*size[2]*size[3]))

-- linear1
--classifier:add (nn.Dropout(0.5))
classifier:add (nn.Linear(size[1]*size[2]*size[3], 160))
classifier:add (nn.ReLU(true))

--linear2
--classifier:add (nn.Dropout(0.5))
classifier:add (nn.Linear(160, 100))
classifier:add (nn.ReLU(true))

classifier:add (nn.Linear(100, nClasses))
--classifier:add (nn.LogSoftMax())

-- combine features and classifiers
local model = nn.Sequential():add(features):add(classifier)

return model

