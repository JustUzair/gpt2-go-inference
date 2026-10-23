package engine

import (
	"fmt"
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
