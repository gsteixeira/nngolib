// NNGolib example
//
package main

import (
    "fmt"
    "github.com/gsteixeira/nngolib"
)

// Main function
func main () {
    // set the parameters for training
    inputs := [][]float64 {{0.0, 0.0},
                           {1.0, 0.0},
                           {0.0, 1.0},
                           {1.0, 1.0}}
    outputs := [][]float64 {{0.0}, {1.0}, {1.0}, {0.0}}
    // instantiate the network
    hidden_sizes := []int {4,}
    // the method can be "sigmoid", "leaky_relu", "relu", and "tanh"
    nn := nngolib.NewNeuralNetwork(len(inputs[0]),
                                   len(outputs[0]),
                                   hidden_sizes,
                                   "leaky_relu")
    // start training
    iteractions := 10000
    nn.Train(inputs, outputs, iteractions)
    // Test trained network
    for i := range inputs {
        predicted := nn.Predict(inputs[i])
        fmt.Println("Input: ", inputs[i],
                    "Expected: ", outputs[i],
                    "Output: ", predicted)
    }
    // dump the network state
    data := nngolib.Dump_nn(nn)
    // You can save that to a file or db.
    // Now create a new network from the saved data
    nngolib.Load_nn(data)
    // make predictions with the saved network
    for i := range inputs {
        predicted := nn.Predict(inputs[i])
        fmt.Println("Input: ", inputs[i],
                    "Expected: ", outputs[i],
                    "Output: ", predicted)
    }
}
