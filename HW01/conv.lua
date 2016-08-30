-- This is the API for 2-d convolution function written in lua

torch.setdefaulttensortype('torch.FloatTensor')
local convLib = {}

function convLib.Lua_conv(x, k)
        local x_row = x:size(1)
	local x_col = x:size(2)
	local k_row = k:size(1)
	local k_col = k:size(2)
	
	local conv_row = x_row-k_row+1
	local conv_col = x_col-k_col+1

	local op = torch.Tensor(conv_row, conv_col)
	
	for row = 1,conv_row do
	    for col = 1,conv_col do
	       op[row][col] = 0;
	       for i = k_row,1,-1 do
		  for j = k_col,1,-1 do
		     op[row][col] = op[row][col] + x[row+(k_row-i)][col+(k_col-j)] * k[i][j]
		  end
	       end
	    end
	end

	return op; 
end

-- This is the API for 2-d convolution function written in C and wrapped by lua

function convLib.C_conv(x,k)
      local x_row = x:size(1)
      local x_col = x:size(2)
      local k_row = k:size(1)
      local k_col = k:size(2)

      local conv_row = x_row-k_row+1
      local conv_col = x_col-k_col+1

      -- FFI stuff -------------------------------------------------------------------
      -- Require ffi
      ffi = require("ffi")
      -- Load myLib
      myLib = ffi.load(paths.cwd() .. '/libconv.so')
      -- Function prototypes definition
      ffi.cdef [[
      void conv(float *op, float *x, float *k, int x_row, int x_col, int k_row, int k_col)
      ]]

      -- Main program ----------------------------------------------------------------
      op = torch.Tensor(conv_row, conv_col)
      myLib.conv(torch.data(op), torch.data(x), torch.data(k), x_row, x_col, k_row, k_col)
      
      -- return the output
      return op;
end

return convLib
