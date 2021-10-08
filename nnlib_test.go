// This is part of the module NNgolib
//
// This tests the basic functionality of the NN
//

package nngolib

import (
    "testing"
)

// Round number IF it's close to 0 or 1
func round(value float64) float64 {
    var rounded float64
    if value > 0.9 {
        rounded = 1
    } else if value < 0.1 {
        rounded = 0
    } else {
        rounded = value
    }
    return rounded
}

// Neural network main test
// This will train the network then check it's capacity to predict
func TestNeuralNetwork(t *testing.T) {
    // set the parameters for training
    inputs := [][]float64 {{0.0, 0.0},
                           {1.0, 0.0},
                           {0.0, 1.0},
                           {1.0, 1.0}}
    outputs := [][]float64 {{0.0}, {1.0}, {1.0}, {0.0}}
    iteractions := 10000

    hidden_sizes := []int {4,}
    nn := NewNeuralNetwork(len(inputs[0]),
                           len(outputs[0]),
                           hidden_sizes,
                           "sigmoid")
    nn.Set_learning_rate(0.1)
    // start training
    nn.Train(inputs, outputs, iteractions)
    // Test trained network
    for i := range inputs {
        predicted := nn.Predict(inputs[i])
        for j := range predicted {
            want := outputs[i][j]
            got := round(predicted[j])
            if got != want {
                t.Errorf("Wrong prediction: %f - %f", got, want)
            }
        }
        t.Log("input: ", inputs[i],
              "predicted:", predicted,
              "output:", outputs[i])
    }
}

// Tests the exporting and inporting of the NN into/from json
func TestNNPersistance(t *testing.T) {
    // declare a generic network
    nn := NewNeuralNetwork(5, 3, []int {4,}, "sigmoid")
    data := Dump_nn(nn)
    nn2 := Load_nn(data)
    for i := range nn.Input_layer.Values {
        if nn.Input_layer.Values[i] != nn2.Input_layer.Values[i] {
            t.Errorf("Got different values!")
        }
    }
    for i := range nn.Input_layer.Bias {
        if nn.Input_layer.Bias[i] != nn2.Input_layer.Bias[i] {
            t.Errorf("Got different values!")
        }
    }
}
