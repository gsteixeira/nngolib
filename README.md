# NNGoLib - A simple Neural Network Go library

NNGolib is a simplistic neural network library for Go.

It aims to be simple as possible, to be easy to use and to allow anyone who wishes to understand neural networks can study it.

NNGolib is built to make it clear as possible. It is done for **educational purposes**.

Yet it provides a working reliable neural network that can be used in your applications with minimal effort.

## Features:

- Support for multiple hidden layers.
- Configurable non linear function. Supports: **sigmoid**, **relu**, **leaky relu** and **tanh**
- (comming soon) Will be possible to persist the network state, so once trained you can store it and use without training again.

## Instalation

```shell
    go get github.com/gsteixeira/nngolib
```

Try the code
```shell
    git clone https://github.com/gsteixeira/nngolib
    cd nngolib
    go mod tidy
    go run sample/sample.go
    go test
```    
    
## usage

```go
    import "github.com/gsteixeira/nngolib"
    ...
    // create the network topology
    nn := nngolib.NewNeuralNetwork(len(inputs[0]),
                                   len(outputs[0]),
                                   []int {4,},
                                   "leaky_relu")
    // train the network
    nn.train(inputs, outputs, iteractions)
    // Make predictions
    predicted = nn.predict(inputs[i])
```

