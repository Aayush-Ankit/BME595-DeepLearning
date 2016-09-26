-- testing the NN
torch.setdefaulttensortype('torch.FloatTensor')

nn = require'NeuralNetwork'

-- Build the network
theta_num = {3,5,3}
nn.build(theta_num)

-- Input to the network
eta = 1
num_inp = 2
in_vec = torch.randn(theta_num[1],num_inp)
print (in_vec)
out_vec1 = in_vec:t()

-- Forward propagate by two approaches
for i = 1,(#theta_num - 1) do
   out_vec1 = torch.cat(out_vec1,torch.ones(out_vec1:size(1),1),2) * nn.getLayer(i)
   out_vec1 = (((out_vec1:mul(-1)):exp()):add(1)):pow(-1)
end
out_vec1 = out_vec1:t();
out_vec2 = nn.forward(in_vec)

-- Check for match of forward propagated values
--print (out_vec1)
--print (out_vec2)
assert((torch.dist(out_vec1, out_vec2) == 0), 'NN Error')

-- Backward propagate for the output labels
label  = torch.Tensor(theta_num[#theta_num], num_inp)
label:random(0,1)
de_dtheta = nn.backward(label)

for i =1,(#de_dtheta) do
   print ("ErrorGradient for Layer:", i)
   print (de_dtheta[i])
end

-- Update Parameters
nn.updateParams(eta)
