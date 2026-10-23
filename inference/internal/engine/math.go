package engine

import (
	"fmt"
	"math"
)

func AddVectors(a, b Tensor) Tensor {
	if len(a) != len(b) {
		panic(fmt.Sprintf("AddVectors: length mismatch %d vs %d", len(a), len(b)))
	}
	result := make(Tensor, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func DotProduct(a, b Tensor) float32 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("DotProduct: length mismatch %d vs %d", len(a), len(b)))
	}
	var result float32
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func MatMul(a Tensor, aRows, aCols int, b Tensor, bRows, bCols int) Tensor {
	if aCols != bRows {
		panic(fmt.Sprintf("MatMul: dimension mismatch %dx%d vs %dx%d", aRows, aCols, bRows, bCols))
	}
	result := make(Tensor, aRows*bCols)

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			for k := 0; k < aCols; k++ {
				result[i*bCols+j] += a[i*aCols+k] * b[k*bCols+j]
			}
		}
	}
	return result
}

// Transpose returns the transpose of a, where a has shape [rows, cols].
// The result has shape [cols, rows].
func Transpose(a Tensor, rows, cols int) Tensor {
	if len(a) != rows*cols {
		panic(fmt.Sprintf("Transpose: len %d != %d*%d", len(a), rows, cols))
	}
	result := make(Tensor, len(a))
	for i := 0; i < cols; i++ { // each row of the result
		for j := 0; j < rows; j++ { // each column of the result
			result[i*rows+j] = a[j*cols+i]
		}
	}
	return result
}

// Softmax: turn a list of raw scores into probabilities (all between 0 and 1,
// adding up to 1). Bigger score = bigger share.
// 1. find the biggest value (max)
// 2. for each value, compute exp(value - max)
// 3. add those up
// 4. divide each one by the total
func SoftMax(input Tensor) Tensor {
	n := len(input)
	if n == 0 {
		panic("Softmax: empty input")
	}
	// step 1: biggest value
	max := input[0]
	for _, v := range input[1:] {
		if v > max {
			max = v
		}
	}

	// step 2 and 3: exp(x - max), and the running total
	output := make(Tensor, n)
	var sum float64
	for i, v := range input {
		e := math.Exp(float64(v - max))
		output[i] = float32(e)
		sum += e
	}

	// step 4: divide by the total so everything adds up to 1
	for i := range output {
		output[i] /= float32(sum)

	}
	return output
}
