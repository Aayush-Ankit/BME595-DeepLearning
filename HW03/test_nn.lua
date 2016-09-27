-- testing the NN
torch.setdefaulttensortype('torch.FloatTensor')
nn = require 'NeuralNetwork'

-- Build the network
theta_num = {2,2,2}
nn.build(theta_num)

-- assigning the thetas from the ijlione example
theta1 = nn.getLayer(1)
theta1[1][1] = 0.15 theta1[1][2] = 0.25
theta1[2][1] = 0.20 theta1[2][2] = 0.30
theta1[3][1] = 0.35 theta1[3][2] = 0.35

theta2 = nn.getLayer(2)
theta2[1][1] = 0.40 theta2[1][2] = 0.50
theta2[2][1] = 0.45 theta2[2][2] = 0.55
theta2[3][1] = 0.60 theta2[3][2] = 0.60

-- Forward
num_inp = 1
in_vec = torch.randn(theta_num[1],num_inp)
in_vec[1][1] = 0.05 in_vec[2][1] = 0.10
in_vec[1][1] = 0.05 in_vec[2][1] = 0.10
out_vec2 = nn.forward(in_vec)
print ("Initial Weights:")
print (nn.getLayer(1))
print (nn.getLayer(2))

-- Backward propagate for the output labels
eta = 0.5
label  = torch.randn(theta_num[#theta_num], num_inp)
label[1] = 0.01 label[2] = 0.99
nn.backward(label)

-- Update Parameters
nn.updateParams(eta)
print ("Initial Weights:")
print (nn.getLayer(1))
print (nn.getLayer(2))
