--Test Script for logic gates

torch.setdefaulttensortype('torch.FloatTensor')
lg = require 'logicGates'

-- Intialize the vectors for testing AND functionality
x = {true, true, false, false} -- standard input and outputs
y = {true, false, true, false}
out_vec_ac = {true, false, false, false}
print ("AND Labels:")
print (out_vec_ac)

--test AND gate for HW02 functionality
--[[lg['AND'].set () -- set the suer assigned values
out_vec_pred_set = {} -- computer the nn output
for i = 1,(#x) do
   out_vec_pred_set[i] = lg['AND'].forward (x[i],y[i])
end
print ("AND Set Results:")
print (out_vec_pred_set)
-- assert ((out_vec_ac == out_vec_pred), 'AND Error') -- equality work only on tensors--]]

--test AND gate's train feature
lg['AND'].train()
out_vec_pred_train = {}
for i = 1,(#x) do
   out_vec_pred_train[i] = lg['AND'].forward (x[i],y[i])
end
print ("AND Trained Results:")
print (out_vec_pred_train)

-- Intialize the vectors for testing OR functionality
x = {true, true, false, false} -- standard input and outputs
y = {true, false, true, false}
out_vec_ac = {true, true, true, false}
print ("OR Labels:")
print (out_vec_ac)

--test OR gate for HW02 functionality
--[[lg['OR'].set () -- set the suer assigned values
out_vec_pred_set = {} -- computer the nn output
for i = 1,(#x) do
   out_vec_pred_set[i] = lg['OR'].forward (x[i],y[i])
end
print ("OR Set Results:")
print (out_vec_pred_set)
-- assert ((out_vec_ac == out_vec_pred), 'OR Error') -- equality work only on tensors--]]

--test OR gate's train feature
lg['OR'].train()
out_vec_pred_train = {}
for i = 1,(#x) do
   out_vec_pred_train[i] = lg['OR'].forward (x[i],y[i])
end
print ("OR Trained Results:")
print (out_vec_pred_train)

-- Intialize the vectors for testing NOT functionality
x = {true, false} -- standard input and outputs
out_vec_ac = {false, true}
print ("NOT Labels:")
print (out_vec_ac)

--test NOT gate for HW02 functionality
--[[lg['NOT'].set () -- set the suer assigned values
out_vec_pred_set = {} -- computer the nn output
for i = 1,(#x) do
   out_vec_pred_set[i] = lg['NOT'].forward (x[i])
end
print ("NOT Set Results:")
print (out_vec_pred_set)
-- assert ((out_vec_ac == out_vec_pred), 'NOT Error') -- equality work only on tensors--]]

--test NOT gate's train feature
lg['NOT'].train()
out_vec_pred_train = {}
for i = 1,(#x) do
   out_vec_pred_train[i] = lg['NOT'].forward (x[i])
end
print ("NOT Trained Results:")
print (out_vec_pred_train)

-- Intialize the vectors for testing XOR functionality
x = {true, true, false, false} -- standard input and outputs
y = {true, false, true, false}
out_vec_ac = {false, true, true, false}
print ("XOR Labels:")
print (out_vec_ac)

--test XOR gate for HW02 functionality
--[[lg['XOR'].set () -- set the suer assigned values
out_vec_pred_set = {} -- computer the nn output
for i = 1,(#x) do
   out_vec_pred_set[i] = lg['XOR'].forward (x[i],y[i])
end
print ("XOR Set Results:")
print (out_vec_pred_set)
-- assert ((out_vec_ac == out_vec_pred), 'OR Error') -- equality work only on tensors--]]

--test OR gate's train feature
lg['XOR'].train()
out_vec_pred_train = {}
for i = 1,(#x) do
   out_vec_pred_train[i] = lg['XOR'].forward (x[i],y[i])
end
print ("XOR Trained Results:")
print (out_vec_pred_train)


