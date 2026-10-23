package engine

import (
	"fmt"
	"math"
)

const (
	epsilon float32 = 0.00001 // 1e-5
)

//		LayerNorm normalizes one token's vector, then applies the learned scale and shift.
//	 1. find the average
//	 2. find how far values are from it, on average (variance)
//	 3. subtract the average and divide by sqrt(variance + epsilon)
//	 4. multiply by weights, add bias (these are the values GPT-2 learned in
//	    training, from the ln_1 / ln_2 / ln_f files)
func LayerNorm(ogEmbeddings, weights, bias Tensor) Tensor {
	var mean float32 = 0
	var variance float32 = 0
	n := len(ogEmbeddings)
	if len(weights) != n || len(bias) != n {
		panic(fmt.Sprintf("LayerNorm: want %d weights and biases, got %d and %d", n, len(weights), len(bias)))
	}
	for _, val := range ogEmbeddings {
		mean += val
	}
	mean /= float32(n)
	for _, val := range ogEmbeddings {
		variance += (val - mean) * (val - mean)
	}
	variance /= float32(n)
	modifiedStandardDeviation := float32(math.Sqrt(float64(variance) + float64(epsilon)))

	output := make(Tensor, n)
	for i := 0; i < n; i++ {
		normalizedValue := (ogEmbeddings[i] - mean) / modifiedStandardDeviation
		output[i] = normalizedValue*weights[i] + bias[i]
	}
	return output
}
