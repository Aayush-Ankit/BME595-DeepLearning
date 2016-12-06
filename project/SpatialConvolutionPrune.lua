local THNN = require 'nn.THNN'
local SpatialConvolutionPrune, parent = torch.class('nn.SpatialConvolutionPrune', 'nn.Module')

function SpatialConvolutionPrune:__init(nInputPlane, nOutputPlane, kW, kH, pruneTH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   --Added for pruning
   self.pruneTH = pruneTH or 0.0
   self.mask = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   --Added for pruning

   self:reset()
end

function SpatialConvolutionPrune:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

--Added for pruning
function SpatialConvolutionPrune:updatepth(pth) -- update the pruneTh
   local temp = self.pruneTH
   self.pruneTH = pth
   --print ("previous th:", temp, "current th", self.pruneTH)
end

function SpatialConvolutionPrune:SeeMap() -- give the mask statistics (#zeros)
   local stats = torch.sum(self.mask) / (self.mask:size(1)*self.mask:size(2)*self.mask:size(3)*self.mask:size(4))
   --print ('Mask Stats: ', stats)
   return stats
end
--Added for pruning

function SpatialConvolutionPrune:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      if self.bias then
         self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then
         self.bias:uniform(-stdv, stdv)
      end
   end
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function SpatialConvolutionPrune:updateOutput(input)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   backCompatibility(self)
   input = makeContiguous(self, input)

   -- added for pruning
   --initialize mask based on current weights
   self.mask = torch.ge(torch.abs(self.weight), self.pruneTH)
   --prune the weights
   self.weight:cmul(self.mask:float())
   -- added for pruning

   input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      THNN.optionalTensor(self.bias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   return self.output
end

function SpatialConvolutionPrune:updateGradInput(input, gradOutput)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   if self.gradInput then
      backCompatibility(self)
      input, gradOutput = makeContiguous(self, input, gradOutput)

      input.THNN.SpatialConvolutionMM_updateGradInput(
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         self.finput:cdata(),
         self.fgradInput:cdata(),
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH
      )
      return self.gradInput
   end
end

function SpatialConvolutionPrune:accGradParameters(input, gradOutput, scale)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   scale = scale or 1
   backCompatibility(self)
   input, gradOutput = makeContiguous(self, input, gradOutput)

   input.THNN.SpatialConvolutionMM_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      THNN.optionalTensor(self.gradBias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      scale
   )
end

function SpatialConvolutionPrune:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function SpatialConvolutionPrune:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   if self.bias then
      return s .. ')'
   else
      return s .. ') without bias'
   end
end

function SpatialConvolutionPrune:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
