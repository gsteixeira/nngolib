// A simple feed forward XOR neural network in Go
//
//   Author: Gustavo Selbach Teixeira
//
//  features:
//    - Allows multiple hidden layers.
//    - configurable non linear method: sigmoid, relu, and leaky_relu.
//    - dump and restore the network data for later use.
// 

package nngolib

import (
    "fmt"
    "math/rand"
)

const VERBOSE = true

// Object that represents the Layer of the network
type Layer struct {
    Values []float64    `json:"values"`
    Bias []float64      `json:"bias"`
    Deltas []float64    `json:"deltas"`
    Weights [][]float64 `json:"weights"`
    N_nodes int         `json:"n_nodes"`
    N_synapses int      `json:"n_synapses"`
}

// Layer object constructor method
func NewLayer (size, parent_size int) Layer {
    layer := Layer {
        Values: make([]float64, size),
        Bias: make([]float64, size),
        Deltas: make([]float64, size),
        Weights: make([][]float64, parent_size),
        N_nodes: size,
        N_synapses: parent_size,
    }
    for i:=0; i<size; i++ {
        layer.Values[i] = rand.Float64()
        layer.Bias[i] = rand.Float64()
    }
    // initialize Weights matrix
    for i := range layer.Weights {
        layer.Weights[i] = make([]float64, size)
    }
    for i:=0; i<size; i++ {
        for j:=0; j<parent_size;j++ {
            layer.Weights[j][i] = rand.Float64()
        }
    }
    return layer
}

// Object that represents the NN. Holds the Layers and settings.
type NeuralNetwork struct {
    Input_layer Layer       `json:"input_layer"`
    Hidden_layer []Layer    `json:"hidden_layers"`
    Output_layer Layer      `json:"output_layer"`
    Learning_rate float64   `json:"learning_rate"`
    Nonlinear_method string `json:"nonlinear_method"`
    nonlinear_function func (float64) float64
    d_nonlinear_function func (float64) float64
}

// The NeuralNetwork constructor method.
func NewNeuralNetwork (input_size,
                       output_size int,
                       hidden_sizes []int,
                       method string,
                      ) NeuralNetwork {
    nn := NeuralNetwork {
        Input_layer: NewLayer(input_size, 0),
        Hidden_layer: make([]Layer, len(hidden_sizes)),
        Learning_rate: 0.05,
        Nonlinear_method: method,
    }
    parent_size := input_size
    for i := range hidden_sizes {
        nn.Hidden_layer[i] = NewLayer(hidden_sizes[i], parent_size)
        parent_size = hidden_sizes[i]
    }
    nn.Output_layer = NewLayer(output_size, parent_size)
    nn.Setup_method(method)
    return nn
}

// Feed inputs to forward through the network
func (nn *NeuralNetwork) Set_inputs (inputs []float64) {
    for i := range inputs {
        nn.Input_layer.Values[i] = inputs[i]
    }
}

// Setup the logistical and derivative function to be used.
func (nn *NeuralNetwork) Setup_method (method string) {
    nn.Nonlinear_method = method
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
}

// Set up the learning rate
func (nn *NeuralNetwork) Set_learning_rate (rate float64) {
    nn.Learning_rate = rate
}

// The activation function
func (nn *NeuralNetwork) activation_function (source, target Layer) {
    var activation float64
    source_length := len(source.Values)
    for j:=0; j<len(target.Values); j++ {
        activation = target.Bias[j]
        for i:=0; i<source_length; i++ {
            activation += (source.Values[i] * target.Weights[i][j])
        }
        target.Values[j] = nn.nonlinear_function(activation)
    }
}

// Calculate the Deltas
func (nn *NeuralNetwork) calc_deltas (source, target Layer) {
    var errors float64
    for j := range target.Values {
        errors = 0.0
        for k := range source.Values {
            errors += (source.Deltas[k] * source.Weights[j][k])
        }
        target.Deltas[j] = (errors * nn.d_nonlinear_function(target.Values[j]))
    }
}

// Calculate the delta for the output layer
func (nn *NeuralNetwork) calc_loss (expected []float64) {
    for i := range nn.Output_layer.Values {
        errors := (expected[i] - nn.Output_layer.Values[i])
        nn.Output_layer.Deltas[i] = (errors * nn.d_nonlinear_function(nn.Output_layer.Values[i]))
    }
}

// Update the Weights of the synapses
func (nn *NeuralNetwork) update_weights (source, target Layer) {
    for j := range source.Values {
        source.Bias[j] += (source.Deltas[j] * nn.Learning_rate)
        for k := range target.Values {
            source.Weights[k][j] += (target.Values[k] * source.Deltas[j] * nn.Learning_rate)
        }
    }
}

// NN Activation step
func (nn *NeuralNetwork) Forward_pass () {
    var j int = 0
    nn.activation_function(nn.Input_layer, nn.Hidden_layer[j])
    for j < len(nn.Hidden_layer)-1 {
        nn.activation_function(nn.Hidden_layer[j],
                            nn.Hidden_layer[j+1])
        j++
    }
    nn.activation_function(nn.Hidden_layer[j], nn.Output_layer)
}

// Backpropagation learning process. Compute the Deltas and update Weights
func (nn *NeuralNetwork) Back_propagation (outputs []float64) {
    var k int
    k = len(nn.Hidden_layer)-1
    // From output layer to the last of the hidden layers
    nn.calc_deltas(nn.Output_layer, nn.Hidden_layer[k])
    nn.update_weights(nn.Output_layer, nn.Hidden_layer[k])
    // Run though the hidden layers. If theres more than 1.
    for k > 0 {
        nn.calc_deltas(nn.Hidden_layer[k], nn.Hidden_layer[k-1])
        nn.update_weights(nn.Hidden_layer[k], nn.Hidden_layer[k-1])
        k -= 1
    }
    // from output to hidden layer
    nn.update_weights(nn.Hidden_layer[k], nn.Input_layer)
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
            if VERBOSE {
                fmt.Println("Input: ", inputs[i],
                            "Expected: ", outputs[i],
                            "Output: ", nn.Output_layer.Values)
            }
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
    return nn.Output_layer.Values
}


// Shuffle array to a random order
func shuffle_array(arr []int) {
    rand.Shuffle(len(arr),
                 func(i, j int) {
                     arr[i], arr[j] = arr[j], arr[i] })
}
