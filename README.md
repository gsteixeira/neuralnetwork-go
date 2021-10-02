# A Neural Network in Go

A feed forward neural network in Go


## usage:

If you just wanna try it:
```shell
    make run
```

## Create a neural network:

Create a network telling the size (nodes) of earch layer.
```go
    nn := NewNeuralNetwork(size_input,
                           size_output,
                           []int {hidden_sizes,})
    // train the network
    nn.Train(inputs, outputs, iteractions)
    // Make predictions
    predicted := nn.Predict(inputs[i])
```
It is possible to use multiple hidden layers. Just pass the size of each one of them in the list:
```go
    // this will create a hidden layer of 3 layers, with sizes 4, 6 and 3.
    nn := NewNeuralNetwork(size_input, size_output,
                           []int {4, 6, 3})
```
## Two implementations:

There are two implementations on this repository:
- neural.go - Object oriented, more flexible and has more features.
- neural_procedural.go - Simple, procedural implentation that uses arrays.



