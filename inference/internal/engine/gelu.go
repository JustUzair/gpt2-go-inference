package engine

import "math"

// Gaussian Error Linear Unit (GELU) activation function

// Exact GELU implementation using the standard library's Error Function
func GELU(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}
