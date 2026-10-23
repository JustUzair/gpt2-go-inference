package repository

import (
	"runtime"
	"strconv"

	"golang.org/x/sync/errgroup"
)

func NewTransformerInput(
	qW, kW, vW Tensor,
	qB, kB, vB Tensor,
	l1W, l2W Tensor,
	l1B, l2B Tensor,
	oW, oB Tensor,
	lnAttnW, lnAttnB Tensor,
	lnMLPW, lnMLPB Tensor,
) *TransformerInput {
	return &TransformerInput{
		QWeights: qW, KWeights: kW, VWeights: vW,
		QBiases: qB, KBiases: kB, VBiases: vB,
		L1Weights: l1W, L2Weights: l2W,
		L1Biases: l1B, L2Biases: l2B,
		OWeights: oW, OBiases: oB,
		LNAttnWeights: lnAttnW, LNAttnBiases: lnAttnB,
		LNMLPWeights: lnMLPW, LNMLPBiases: lnMLPB,
	}
}

// loadLayer fills one transformer layer from its files.
func loadLayer(layer *TransformerInput, prefix string) error {
	// ln_1 (normalizes the input before attention)
	if err := load(&layer.LNAttnWeights, prefix+".ln_1.weight.txt"); err != nil {
		return err
	}
	if err := load(&layer.LNAttnBiases, prefix+".ln_1.bias.txt"); err != nil {
		return err
	}

	// one fused file holds Q, K and V together; this splits it into three
	if err := loadFusedQKV(layer, prefix); err != nil {
		return err
	}

	// attention output projection
	if err := load(&layer.OWeights, prefix+".attn.c_proj.weight.txt"); err != nil {
		return err
	}
	if err := load(&layer.OBiases, prefix+".attn.c_proj.bias.txt"); err != nil {
		return err
	}

	// ln_2 (normalizes the input before the MLP)
	if err := load(&layer.LNMLPWeights, prefix+".ln_2.weight.txt"); err != nil {
		return err
	}
	if err := load(&layer.LNMLPBiases, prefix+".ln_2.bias.txt"); err != nil {
		return err
	}

	// mlp: expand layer (l1), then shrink layer (l2)
	if err := load(&layer.L1Weights, prefix+".mlp.c_fc.weight.txt"); err != nil {
		return err
	}
	if err := load(&layer.L1Biases, prefix+".mlp.c_fc.bias.txt"); err != nil {
		return err
	}
	if err := load(&layer.L2Weights, prefix+".mlp.c_proj.weight.txt"); err != nil {
		return err
	}
	return load(&layer.L2Biases, prefix+".mlp.c_proj.bias.txt")
}

func NewGPTWeights() (*GPTWeights, error) {
	g := &GPTWeights{
		TransformerWeights: make([]TransformerInput, numLayers),
	}

	// eg runs functions in parallel and remembers the first error.
	// The limit stops us from running more at once than we have CPU cores.
	var eg errgroup.Group
	eg.SetLimit(runtime.NumCPU())

	// Start the biggest file first (about 38M numbers) so it isn't left until last.
	eg.Go(func() error {
		return load(&g.EmbeddingWeights, weightsDir+"transformer.wte.weight.txt")
	})
	eg.Go(func() error {
		return load(&g.PositionalEmbeddingWeights, weightsDir+"transformer.wpe.weight.txt")
	})

	// One task per layer. Each task touches only its own layer.
	for i := 0; i < numLayers; i++ {
		eg.Go(func() error {
			prefix := weightsDir + "transformer.h." + strconv.Itoa(i)
			return loadLayer(&g.TransformerWeights[i], prefix)
		})
	}

	// final layernorm, applied once after all 12 layers
	eg.Go(func() error {
		return load(&g.FinalWeights, weightsDir+"transformer.ln_f.weight.txt")
	})
	eg.Go(func() error {
		return load(&g.FinalBias, weightsDir+"transformer.ln_f.bias.txt")
	})

	// Wait is required: without it we'd return before the loading finishes.
	if err := eg.Wait(); err != nil {
		return nil, err
	}
	return g, nil
}
