// This is part of the module NNgolib
//
// This file is responsible for dumping and loading
// NeuralNetwork data into/from json

package nngolib

import "encoding/json"

// Takes a NeuralNetwork object and serialize it to json
func Dump_nn(nn NeuralNetwork) string {
    json_data, err := json.Marshal(nn)
    if err != nil { panic(err) }
    return string(json_data)
}

// Takes a json representing a NN and load it into NeuralNetwork object
func Load_nn(json_data string) NeuralNetwork {
    var nn NeuralNetwork
    err := json.Unmarshal([]byte(json_data), &nn)
    if err != nil { panic(err) }
    nn.Setup_method(nn.Nonlinear_method)
    return nn
}

