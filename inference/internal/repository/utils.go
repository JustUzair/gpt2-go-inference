package repository

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

const (
	numLayers  = 12
	nEmb       = 768
	nFused     = 3 * nEmb // Q | K | V side by side
	weightsDir = "../weights/"
)

// newScanner makes a scanner that returns one number (word) at a time.
// Split is set here, once, because Go doesn't allow changing it after reading starts.
func newScanner(f *os.File) *bufio.Scanner {
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	return scanner
}

// Weights Utils

// readFromFileStream reads numbers from the scanner and appends them to vec.
// maxCount > 0: read exactly that many numbers (error if the file runs out early).
// maxCount <= 0: read everything left in the file.
// Calling it again on the same scanner carries on where the last call stopped.
func readFromFileStream(vec *Tensor, scanner *bufio.Scanner, maxCount int) error {
	count := 0
	for (maxCount <= 0 || count < maxCount) && scanner.Scan() {
		val, err := strconv.ParseFloat(scanner.Text(), 32)
		if err != nil {
			return err
		}
		*vec = append(*vec, float32(val))
		count++
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if maxCount > 0 && count < maxCount {
		return fmt.Errorf("file ended early: wanted %d numbers, got %d", maxCount, count)
	}
	return nil
}

// load reads a whole file into vec.
func load(vec *Tensor, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	if err := readFromFileStream(vec, newScanner(f), 0); err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	return nil
}

// loadQKVBiases reads the fused bias file. It holds 2304 numbers in order:
// first 768 are Q, next 768 are K, last 768 are V.
func loadQKVBiases(layer *TransformerInput, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	// One scanner, three reads: each read continues where the previous one stopped.
	scanner := newScanner(f)
	for _, dst := range []*Tensor{&layer.QBiases, &layer.KBiases, &layer.VBiases} {
		if err := readFromFileStream(dst, scanner, nEmb); err != nil {
			return fmt.Errorf("read %s: %w", path, err)
		}
	}
	return nil
}

// loadQKVWeights reads the fused weight file. It has 768 rows, and every row
// holds 2304 numbers: 768 for Q, then 768 for K, then 768 for V.
// So we read one row at a time and send each third to its own tensor.
func loadQKVWeights(layer *TransformerInput, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	scanner := newScanner(f)

	layer.QWeights = make(Tensor, 0, nEmb*nEmb)
	layer.KWeights = make(Tensor, 0, nEmb*nEmb)
	layer.VWeights = make(Tensor, 0, nEmb*nEmb)

	for r := 0; r < nEmb; r++ {
		row := make(Tensor, 0, nFused)
		if err := readFromFileStream(&row, scanner, nFused); err != nil {
			return fmt.Errorf("read %s row %d: %w", path, r, err)
		}
		layer.QWeights = append(layer.QWeights, row[0:nEmb]...)
		layer.KWeights = append(layer.KWeights, row[nEmb:2*nEmb]...)
		layer.VWeights = append(layer.VWeights, row[2*nEmb:nFused]...)
	}
	return nil
}

// loadFusedQKV fills Q, K and V (weights and biases) from the two fused files.
func loadFusedQKV(layer *TransformerInput, prefix string) error {
	if err := loadQKVBiases(layer, prefix+".attn.c_attn.bias.txt"); err != nil {
		return err
	}
	return loadQKVWeights(layer, prefix+".attn.c_attn.weight.txt")
}

// Vocab Utils

func parseMerges() {}
