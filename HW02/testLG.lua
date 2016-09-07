--Test Script for logic gates

torch.setdefaulttensortype('torch.FloatTensor')
lg = require'logicGates'

--create input vector
x = {true, true, false, false}
y = {true, false, true, false}

--test and gate
out_vec_ac = {true, false, false, false}
out_vec_pred = {}
for i = 1,(#x) do
   out_vec_pred[i] = lg.AND(x[i],y[i])
end
print (out_vec_ac)
print (out_vec_pred)

--test or gate
out_vec_ac = {true, true, true, false}
out_vec_pred = {}
for i = 1,(#x) do
   out_vec_pred[i] = lg.OR(x[i],y[i])
end
print (out_vec_ac)
print (out_vec_pred)

--test xor gate
out_vec_ac = {false, true, true, false}
out_vec_pred = {}
for i = 1,(#x) do
   out_vec_pred[i] = lg.XOR(x[i],y[i])
end
print (out_vec_ac)
print (out_vec_pred)

--test not gate
out_vec_ac = {false, false, true, true}
out_vec_pred = {}
for i = 1,(#x) do
   out_vec_pred[i] = lg.NOT(x[i])
end
print (out_vec_ac)
print (out_vec_pred)
