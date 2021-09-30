// Python implementation of a simple Feedforward Neural Network
//
// Author: Gustavo Selbach Teixeira (gsteixei@gmail.com)
//
// Procedural simplistic implementation. 
// Uses only arrays instead of objects
//
package main
import (
    "fmt"
    "math"
    "math/rand"
)
// Set parameters and call training
func main () {
    // set the parameters for training
    inputs := [][]float64 {{0.0, 0.0},
                            {1.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 1.0}}
    outputs := [][]float64 {{0.0}, {1.0}, {1.0}, {0.0}}
    iteractions := 100000
    train(inputs, outputs, iteractions)
}

// The logistical sigmoid function
func sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

// The derivative of sigmoid function
func d_sigmoid(x float64) float64 {
    return x * (1 - x)
}

// Initialize the vectors for a new layer
func new_layer(size int, parent_size int) ([]float64, []float64, [][]float64) {
    layer := make([]float64, size)
    layer_bias := make([]float64, size)
    layer_weights := make([][]float64, parent_size)
    
    for i:=0; i<size; i++ {
        layer[i] = rand.Float64()
        layer_bias[i] = rand.Float64()
    }
    // initialize weights matrix
    for i := range layer_weights {
        layer_weights[i] = make([]float64, size)
    }
    for i:=0; i<size; i++ {
        for j:=0; j<parent_size;j++ {
            layer_weights[j][i] = rand.Float64()
        }
    }
    return layer, layer_bias, layer_weights
}

// The Activation function
func activation_function(source_layer,
                         target_layer,
                         target_layer_bias []float64,
                         target_weights [][]float64) {
    var activation float64
    var source_length int
    source_length = len(source_layer)
    
    for j:=0; j<len(target_layer); j++ {
        activation = target_layer_bias[j]
        for i:=0; i<source_length; i++ {
            
            activation += (source_layer[i] * target_weights[i][j])
        }
        target_layer[j] = sigmoid(activation)
    }
}

// Calculate the delta for the output layer
func calc_delta_output(expected []float64, output_layer []float64) []float64 {
    delta_output := make([]float64, len(output_layer))
    for i := range output_layer {
        error_output := (expected[i] - output_layer[i])
        delta_output[i] = (error_output * d_sigmoid(output_layer[i]))
    }
    return delta_output
}

// Calculate the Deltas
func calc_deltas(source_layer,
                source_delta,
                target_layer []float64,
                source_weights [][]float64) []float64{
    var errors float64
    delta := make([]float64, len(target_layer))
    for j := range target_layer {
        errors = 0.0
        for k := range source_layer {
            errors += (source_delta[k] * source_weights[j][k])
        }
        delta[j] = (errors * d_sigmoid(target_layer[j]))
    }
    return delta
}

// Update the weights of the synapses
func update_weights(source_bias []float64,
                   source_weights [][]float64,
                   source_delta []float64,
                   target_layer []float64,
                   learning_rate float64) {
    for j := range source_bias {
        source_bias[j] += (source_delta[j] * learning_rate)
        for k := range target_layer {
            source_weights[k][j] += (target_layer[k] * source_delta[j] * learning_rate)
        }
    }
}

// Shuffle array to a random order
func shuffle_array(arr []int) {
    rand.Shuffle(len(arr),
                 func(i, j int) {
                     arr[i], arr[j] = arr[j], arr[i] })
}

// Neural network training loop
func train(training_inputs, training_outputs [][]float64, n_epochs int){
    var i int
    num_inputs := len(training_inputs[0])
    num_outputs := len(training_outputs[0])
    num_training_sets := len(training_inputs)
    
    num_hidden_nodes := 32
    learning_rate := 0.1
    
    // Initialize the vectors for the hidden layer
    hidden_layer, hidden_bias, hidden_weights := new_layer(
        num_hidden_nodes, num_inputs)

    // Initialize the vectors for the output layer
    output_layer, output_layer_bias, output_weights := new_layer(
        num_outputs, num_hidden_nodes)
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
            activation_function(training_inputs[i],
                                hidden_layer,
                                hidden_bias,
                                hidden_weights)
            activation_function(hidden_layer,
                                output_layer,
                                output_layer_bias,
                                output_weights)
            // Show results
            fmt.Println("Input: ", training_inputs[i],
                        "Expected: ", training_outputs[i],
                        "Output: ", output_layer[0])
            // Learning
            // calculate the deltas
            delta_output := calc_delta_output(training_outputs[i], output_layer)
            delta_hidden := calc_deltas(output_layer,
                                        delta_output,
                                        hidden_layer,
                                        output_weights)
            // from output to hidden layer
            update_weights(output_layer_bias,
                           output_weights,
                           delta_output,
                           hidden_layer,
                           learning_rate)
            // from hidden to input layer
            update_weights(hidden_bias,
                           hidden_weights,
                           delta_hidden,
                           training_inputs[i],
                           learning_rate)
        }
    }
}
