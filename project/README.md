# Title
Exploring Neural Network Pruning for Energy-Efficient Hardware realizations

## Team members
Aayush Ankit (Aayush-Ankit), Chankyu Lee

### Goals
ANalyze the effect of network pruning on neural network's accuracy, training effort and parameter reduction.

## Challenges
Coming up with modules - nn.Prune to add a pruining layer in the ConvNet and MLPs

### nn.LinearPrune is the nn.Linear & nn.Prune modules combined together.
### nn.SpatialConvolutionPrune is the nn.SpatialConvolution & nn.Prune modules conbined together.

### img2num.lua - script to train MNIST on different models. The script also contains the prune contraol script.
### img2obj.lua - script to train CIFAR-10 on different models. The script also contains the prune contraol script.
