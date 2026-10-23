package main

import (
	"engine/internal/repository"
	"fmt"
	"log"
	"time"
)

func main() {
	// load weights
	start := time.Now()
	weights, err := repository.NewGPTWeights()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("loaded in", time.Since(start))

	// get embedding
	repository.LoadVocab()
}
