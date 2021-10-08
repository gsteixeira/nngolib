# NNGoLib - A simple Neural Network Go library

NNGolib is a simplistic neural network library for Go.

It aims to be simple as possible, to be easy to use and to allow anyone who wishes to understand neural networks can study it.

NNGolib is built to make it clear as possible. It is done for **educational purposes**.

Yet it provides a working reliable neural network that can be used in your applications with minimal effort.

## Features:

- Support for multiple hidden layers.
- Configurable non linear function. Supports: **sigmoid**, **relu**, **leaky relu** and **tanh**
- It is possible to persist the network state, so once trained you can store it and use without training again.

## Instalation

If you want to use it:

```shell
    go get github.com/gsteixeira/nngolib
```

If you want to play with it:

```shell
    git clone https://github.com/gsteixeira/nngolib
    cd nngolib
    go mod tidy
    go run sample/sample.go
    go test
```

## usage

Create a Neural Network by telling the sizes of *input layer*, the *output layer*, and a list with the sizes of each of the *hidden layers*. Usually you will only need one, so declare an array of one position.

```go
    import "github.com/gsteixeira/nngolib"
    ...
    input_size := len(input_data[0]) // the size of input layer
    output_size := len(expected_output[0]) // the size of output layer
    hidden_layers := []int {4,} // the sizes of each of the hidden layers
    // create the network topology
    nn := nngolib.NewNeuralNetwork(input_size,
                                   output_size,
                                   hidden_layers,
                                   "leaky_relu")
    // train the network (10 thousand times)
    nn.train(input_data, expected_output, 10000)
    // Make predictions
    predicted = nn.predict(input_data[x])
```

Now you can save your network's state and reuse it later.

```go
    // dump the network state
    data := nngolib.Dump_nn(nn)
    // You can save that to a file or db.
    // Now create a new network from the saved data
    nn2 := nngolib.Load_nn(data)
    // make predictions with the saved network
    predicted = nn.predict(foo)
```
