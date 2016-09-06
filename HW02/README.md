### ***_ I am not using a local table inside the NeuralNetwork API , rather declare a global table in the script that calls the NN API and the logicGates API _***
#HW02 - NeuralNetwork (NN) API and LogicGate (LG) API

## NN API has three functions build(), getLayer() and forward().
## LG API has four functions AND, OR, NOT, XOR.

### NN API functions :-
1. build() -  builds up a neural network of given size - no. of layers, size of each layer. Returns a table of theta matrices.
2. getLayer() - returns the theta matrix corresponding lo **layer(i)** and **layer(i+1)**
3. forward() - propagates input across the neural network and returns the final output vector.

### LG API :- (takes boolean inputs and returns boolean output)
1. AND - logical **AND** of two inputs
2. OR - logical **OR** of two inputs
3. NOT - logical **NOT** of an inputs
4. XOR - logical **XOR** of two inputs
