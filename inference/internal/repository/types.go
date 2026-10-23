package repository

import "encoding/json"

// Per-layer weights for GPT-2 small. d = 768, dFF = 3072.
// Layout is [in, out], so y = x·W + b.
// Source files: transformer.h.{i}.<tensor>.txt
//
// ┌───────────────┬────────────┬──────────────────────────────────┬───────────┐
// │ Go field      │ Shape      │ GPT-2 tensor                     │ Floats    │
// ├───────────────┼────────────┼──────────────────────────────────┼───────────┤
// │ QWeights      │ [768, 768] │ attn.c_attn.weight[:, 0:768]     │ 589,824   │
// │ KWeights      │ [768, 768] │ attn.c_attn.weight[:, 768:1536]  │ 589,824   │
// │ VWeights      │ [768, 768] │ attn.c_attn.weight[:, 1536:2304] │ 589,824   │
// │ QBiases       │ [768]      │ attn.c_attn.bias[0:768]          │ 768       │
// │ KBiases       │ [768]      │ attn.c_attn.bias[768:1536]       │ 768       │
// │ VBiases       │ [768]      │ attn.c_attn.bias[1536:2304]      │ 768       │
// ├───────────────┼────────────┼──────────────────────────────────┼───────────┤
// │ L1Weights     │ [768, 3072]│ mlp.c_fc.weight                  │ 2,359,296 │
// │ L2Weights     │ [3072, 768]│ mlp.c_proj.weight                │ 2,359,296 │
// │ L1Biases      │ [3072]     │ mlp.c_fc.bias                    │ 3,072     │
// │ L2Biases      │ [768]      │ mlp.c_proj.bias                  │ 768       │
// ├───────────────┼────────────┼──────────────────────────────────┼───────────┤
// │ OWeights      │ [768, 768] │ attn.c_proj.weight               │ 589,824   │
// │ OBiases       │ [768]      │ attn.c_proj.bias                 │ 768       │
// ├───────────────┼────────────┼──────────────────────────────────┼───────────┤
// │ LNAttnWeights │ [768]      │ ln_1.weight (gamma)              │ 768       │
// │ LNAttnBiases  │ [768]      │ ln_1.bias (beta)                 │ 768       │
// │ LNMLPWeights  │ [768]      │ ln_2.weight (gamma)              │ 768       │
// │ LNMLPBiases   │ [768]      │ ln_2.bias (beta)                 │ 768       │
// └───────────────┴────────────┴──────────────────────────────────┴───────────┘
//
// Q, K and V all come from the single fused c_attn file.
// The slice [:, a:b] means columns a to b of every row, not a contiguous chunk.

type Tensor = []float32

type TransformerInput struct {

	// Attention weights and biases
	QWeights, KWeights, VWeights Tensor
	QBiases, KBiases, VBiases    Tensor

	// MLP weights and biases
	L1Weights, L2Weights Tensor
	L1Biases, L2Biases   Tensor

	OWeights, OBiases Tensor

	// LayerNorm weights and biases
	// Before feeding into attention or MLP, we apply layer normalization to the input
	LNAttnWeights, LNAttnBiases Tensor
	LNMLPWeights, LNMLPBiases   Tensor
}

type GPTWeights struct {
	// Embeddings, Positional Embeddings
	EmbeddingWeights, PositionalEmbeddingWeights Tensor
	// Transformer weights and biases for each layer
	TransformerWeights      []TransformerInput
	FinalBias, FinalWeights Tensor
}

type Tokenizer struct {
	Tokens    []string          // id -> string
	TokenToID map[string]int    // token string -> id
	Merges    map[[2]string]int // (left, right) -> rank; lower rank merges first
}
type TokenizerFile struct {
	Model struct {
		Vocab  map[string]int    `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
}
