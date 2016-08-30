  local convLib = require 'conv'
  local kMaxSize = 5
  local imgMaxSize = 100
  local r = math.random
  local imgA = torch.rand(kMaxSize + r(imgMaxSize), kMaxSize + r(imgMaxSize))
  local imgB = torch.rand(r(kMaxSize), r(kMaxSize))
  local torchConv = torch.conv2(imgA, imgB)
  local luaConv = convLib.Lua_conv(imgA, imgB)
  local cConv = convLib.C_conv(imgA, imgB)
   
  --measures the distance between user results and golden results
  assert(torch.dist(torchConv, luaConv) < 1e-3, 'Lua Error')
  assert(torch.dist(torchConv, cConv) < 1e-3, 'C Error')
