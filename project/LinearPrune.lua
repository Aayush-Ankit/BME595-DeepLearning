local LinearPrune, parent = torch.class('nn.LinearPrune', 'nn.Module')

function LinearPrune:__init(inputSize, outputSize, pruneTh, bias) -- added a 4th parameter for pruning
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end

   -- modified - Aayush Ankit - add a pruneMap Tensor
   self.pruneTH = pruneTH or 0.0
   self.mask = torch.Tensor(outputSize, inputSize)
   -- modified - Aayush Ankit - add a pruneMap Tensor

   self:reset()
end

-- modified - Aayush Ankit - add a fucntion to update the prume threshold
function LinearPrune:updatepth(pth)
   local temp = self.pruneTH
   self.pruneTH = pth
   --print ("previous th:", temp, "current th", self.pruneTH)
end
function LinearPrune:SeeMap() -- give the mask statistics (#zeros)
   local stats = torch.sum(self.mask) / (self.mask:size(1)*self.mask:size(2))
   --print ('Mask Stats: ', stats)
   return stats
end
-- modified - Aayush Ankit

function LinearPrune:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function LinearPrune:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

local function updateAddBuffer(self, input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

function LinearPrune:updateOutput(input)

   -- added for pruning
   --initialize mask based on current weights
   self.mask = torch.ge(torch.abs(self.weight), self.pruneTH)
   --prune the weights
   self.weight:cmul(self.mask:float())
   -- added for pruning

   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      updateAddBuffer(self, input)
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearPrune:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero() -- doesnt matter if pruned weights become zero
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function LinearPrune:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      if self.bias then
         -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
         updateAddBuffer(self, input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
end

-- we do not need to accumulate parameters when sharing
LinearPrune.sharedAccUpdateGradParameters = LinearPrune.accUpdateGradParameters

function LinearPrune:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function LinearPrune:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end


