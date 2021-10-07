// This is part of the module NNgolib
//
// The non linear, or logistical functions and their derivatives
//

package nngolib

import (
    "math"
)

// The logistical sigmoid function
func sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

// The derivative of sigmoid function
func d_sigmoid(x float64) float64 {
    return x * (1 - x)
}

// The logistical Leaky Relu function
func leaky_relu(x float64) float64 {
    if x > 0 {
        return x
    } else {
        return x * 0.01
    }
}

// The derivative of Leaky Relu function
func d_leaky_relu(x float64) float64 {
    if x >= 0 {
        return 1.0
    } else {
        return 0.01
    }
}

// The logistical Leaky Relu function
func relu(x float64) float64 {
    if x > 0 {
        return x
    } else {
        return 0
    }
        
}

// The derivative of Leaky Relu function
func d_relu(x float64) float64 {
    if x >= 0 {
        return 1.0
    } else {
        return 0
    }
}

// The Tanh function
func tanh(x float64) float64 {
    return math.Tanh(x)
}

// The derivative of Tanh function
func d_tanh(x float64) float64 {
    return 1 - math.Pow(math.Tanh(x), 2)
}
