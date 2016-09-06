-- testing the NN
torch.setdefaulttensortype('torch.FloatTensor')
tbl = {}

a = require'NeuralNetwork'

--theta_num = torch.Tensor({3,2,2})
theta_num = {5,10,3}

a.build(theta_num)

in_vec = torch.Tensor(theta_num[1],8)
out_vec1 = in_vec:t()
--for i = 1,(theta_num:size(1)-1) do
for i = 1,(#theta_num - 1) do
   out_vec1 = torch.cat(out_vec1,torch.ones(out_vec1:size(1),1),2) * a.getLayer(i)
   out_vec1 = (((out_vec1:mul(-1)):exp()):add(1)):pow(-1)
end
out_vec2 = a.forward(in_vec)

print (out_vec1)
print (out_vec2)
value = (torch.dist(out_vec1, out_vec2) == 0) and 'True' or 'False'
print(value)
--assert(torch.dist(out_vec1, out_vec2) < 1e-3, 'NN Error')
