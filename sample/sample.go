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
                                   "tanh")
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
}
