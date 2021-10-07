// A simple feed forward XOR neural network in Go
//
//  features:
//    - Allows multiple hidden layers
//    - configurable non linear method: sigmoid, relu, and leaky_relu
// 
// Author: Gustavo Selbach Teixeira
//

package nngolib

import (
    "log"
    "math/rand"
)

type Layer struct {
    values []float64 `json:"values"`
    bias []float64   `json:"bias"`
    deltas []float64 `json:"deltas"`
    weights [][]float64 `json:"weights"`
    n_nodes int         `json:"n_nodes"`
    n_synapses int      `json:"n_synapses"`
}

func NewLayer (size, parent_size int) Layer {
    layer := Layer {
        values: make([]float64, size),
        bias: make([]float64, size),
        deltas: make([]float64, size),
        weights: make([][]float64, parent_size),
        n_nodes: size,
        n_synapses: parent_size,
    }
    for i:=0; i<size; i++ {
        layer.values[i] = rand.Float64()
        layer.bias[i] = rand.Float64()
    }
    // initialize weights matrix
    for i := range layer.weights {
        layer.weights[i] = make([]float64, size)
    }
    for i:=0; i<size; i++ {
        for j:=0; j<parent_size;j++ {
            layer.weights[j][i] = rand.Float64()
        }
    }
    return layer
}

type LogFunc func (float64) float64

type NeuralNetwork struct {
    input_layer Layer `json:"input_layer"`
    hidden_layer []Layer `json:"hidden_layer"`
    output_layer Layer `json:"output_layer"`
    learning_rate float64 `json:"output_layer"`
    
    nonlinear_method string `json:"output_layer"`
    nonlinear_function LogFunc
    d_nonlinear_function LogFunc
}

// Create a new Neural Network
func NewNeuralNetwork (input_size,
                       output_size int,
                       hidden_sizes []int,
                       method string,
                      ) NeuralNetwork {
    nn := NeuralNetwork {
        input_layer: NewLayer(input_size, 0),
        hidden_layer: make([]Layer, len(hidden_sizes)),
        learning_rate: 0.05,
        nonlinear_method: method,
    }
    parent_size := input_size
    for i := range hidden_sizes {
        nn.hidden_layer[i] = NewLayer(hidden_sizes[i], parent_size)
        parent_size = hidden_sizes[i]
    }
    nn.output_layer = NewLayer(output_size, parent_size)
    switch method {
        case "leaky_relu":
            nn.nonlinear_function = leaky_relu
            nn.d_nonlinear_function = d_leaky_relu
        case "sigmoid":
            nn.nonlinear_function = sigmoid
            nn.d_nonlinear_function = d_sigmoid
        case "relu":
            nn.nonlinear_function = relu
            nn.d_nonlinear_function = d_relu
        case "tanh":
            nn.nonlinear_function = tanh
            nn.d_nonlinear_function = d_tanh
        default:
            panic("Invalid method: " + method)
    }
    return nn
}
                           
// Feed inputs to forward through the network
func (nn *NeuralNetwork) Set_inputs (inputs []float64) {
    for i := range inputs {
        nn.input_layer.values[i] = inputs[i]
    }
}
            
// Set up the learning rate
func (nn *NeuralNetwork) Set_learning_rate (rate float64) {
    nn.learning_rate = rate
}

// The activation function
func (nn *NeuralNetwork) activation_function (source, target Layer) {
    var activation float64
    source_length := len(source.values)
    for j:=0; j<len(target.values); j++ {
        activation = target.bias[j]
        for i:=0; i<source_length; i++ {
            activation += (source.values[i] * target.weights[i][j])
        }
        target.values[j] = nn.nonlinear_function(activation)
    }
}

// Calculate the Deltas
func (nn *NeuralNetwork) calc_deltas (source, target Layer) {
    var errors float64
    for j := range target.values {
        errors = 0.0
        for k := range source.values {
            errors += (source.deltas[k] * source.weights[j][k])
        }
        target.deltas[j] = (errors * nn.d_nonlinear_function(target.values[j]))
    }
}

// Calculate the delta for the output layer
func (nn *NeuralNetwork) calc_loss (expected []float64) {
    for i := range nn.output_layer.values {
        errors := (expected[i] - nn.output_layer.values[i])
        nn.output_layer.deltas[i] = (errors * nn.d_nonlinear_function(nn.output_layer.values[i]))
    }
}

// Update the weights of the synapses
func (nn *NeuralNetwork) update_weights (source, target Layer) {
    for j := range source.values {
        source.bias[j] += (source.deltas[j] * nn.learning_rate)
        for k := range target.values {
            source.weights[k][j] += (target.values[k] * source.deltas[j] * nn.learning_rate)
        }
    }
}

// NN Activation step
func (nn *NeuralNetwork) Forward_pass () {
    var j int = 0
    nn.activation_function(nn.input_layer, nn.hidden_layer[j])
    for j < len(nn.hidden_layer)-1 {
        nn.activation_function(nn.hidden_layer[j],
                            nn.hidden_layer[j+1])
        j++
    }
    nn.activation_function(nn.hidden_layer[j], nn.output_layer)
}

// Backpropagation learning process
func (nn *NeuralNetwork) Back_propagation (outputs []float64) {
    var last_hidden_layer, k int
    // calculate the deltas
    last_hidden_layer = len(nn.hidden_layer)-1
    k = last_hidden_layer
    // From output layer to the last of the hidden layers
    nn.calc_deltas(nn.output_layer, nn.hidden_layer[k])
    // Run though the hidden layers. If theres more than 1.
    for k > 0 {
        nn.calc_deltas(nn.hidden_layer[k], nn.hidden_layer[k-1])
        k -= 1
    }
    // Update weights and bias
    k = last_hidden_layer
    nn.update_weights(nn.output_layer, nn.hidden_layer[k])
    for k > 0 {
        nn.update_weights(nn.hidden_layer[k], nn.hidden_layer[k-1])
        k -= 1
    }
    // from output to hidden layer
    nn.update_weights(nn.hidden_layer[k], nn.input_layer)
}

// Train the neural network
func (nn *NeuralNetwork) Train (inputs, outputs [][]float64, n_epochs int) {
    var i int
    num_training_sets := len(inputs)
    // randomize training to avoid network to become grandmothered
    training_sequence := make([]int, num_training_sets)
    for i=0; i<num_training_sets; i++ {
        training_sequence[i] = i
    }
    for n:=0; n<n_epochs; n++ {
        shuffle_array(training_sequence)
        for x:=0; x<num_training_sets; x++ {
            i := training_sequence[x]
            // Forward pass
            nn.Set_inputs(inputs[i])
            nn.Forward_pass()
            // Show results
            log.Println("Input: ", inputs[i],
                        "Expected: ", outputs[i],
                        "Output: ", nn.output_layer.values)
            // Learning
            nn.calc_loss(outputs[i])
            nn.Back_propagation(outputs[i])
        }
    }
}

// Make a prediction. To be used once the network is trained
func (nn *NeuralNetwork) Predict (inputs []float64) []float64{
    nn.Set_inputs(inputs)
    nn.Forward_pass()
    return nn.output_layer.values
}


// Shuffle array to a random order
func shuffle_array(arr []int) {
    rand.Shuffle(len(arr),
                 func(i, j int) {
                     arr[i], arr[j] = arr[j], arr[i] })
}
