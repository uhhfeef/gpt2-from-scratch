"""GPT-2 124M, forward pass and greedy sampling, written from scratch in PyTorch.

1. define GPT_CONFIG_124M with the real 124M shapes -> vocab 50257, 12 layers, 768 hidden
2. build causal self-attention, first single-head then split across 12 heads
3. stack 12 pre-norm blocks of attention + GELU MLP, each wrapped in a residual
4. project the final hidden states to vocabulary logits through lm_head
5. run under __main__: encode a prompt, generate 6 tokens, decode -> printed text

Weights are randomly initialised, so the decoded text is nonsense. This file is
about the architecture, not about a trained model.
"""

import tiktoken
import torch
import torch.nn as nn

torch.manual_seed(42)

GPT_CONFIG_124M = {
    "vocab_size": 50257,             # Vocabulary size
    "max_position_embeddings": 1024, # Context length
    "hidden_size": 768,              # Embedding dimension
    "num_attention_heads": 12,       # Number of attention heads
    "num_hidden_layers": 12,         # Number of layers
    "attention_dropout": 0.1,        # Dropout rate
    "attention_bias": False          # Query-Key-Value bias
}


class CausalAttention(nn.Module):
    """Single-head self-attention that cannot look at future tokens."""

    def __init__(self, hidden_size, max_position_embeddings,
                 attention_dropout, attention_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.dropout = nn.Dropout(attention_dropout)
        # Upper-triangular = the positions each token is NOT allowed to see.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_position_embeddings, max_position_embeddings), diagonal=1),
        )

    def forward(self, x):
        """Turns token embeddings into context-aware embeddings.

        1. project x into queries, keys and values
        2. dot every query against every key -> attn_scores (B, T, T)
        3. fill the future positions with -inf so softmax sends them to 0
        4. scale by sqrt(hidden_size) and softmax -> attn_weights
        5. weight the values by attn_weights -> returned attn_output
        """
        batch_size, seq_len, _ = x.shape
        key_states = self.k_proj(x)
        query_states = self.q_proj(x)
        value_states = self.v_proj(x)

        attn_scores = query_states @ key_states.transpose(1, 2)
        attn_scores.masked_fill_(
            self.causal_mask.bool()[:seq_len, :seq_len], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / key_states.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ value_states
        return attn_output


class MultiheadAttention(nn.Module):
    """Causal self-attention run in parallel across num_heads subspaces."""

    def __init__(self, hidden_size, num_heads, max_position_embeddings,
                 attention_dropout, attention_bias=False):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # One projection each for queries, keys and values.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)

        # Mixes the heads back together after they are concatenated.
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_dropout)

        # Upper-triangular = the positions each token is NOT allowed to see.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_position_embeddings, max_position_embeddings), diagonal=1).bool(),
        )

    def forward(self, x):
        """Turns token embeddings into context-aware embeddings, per head.

        1. project x into queries, keys and values
        2. reshape each to (B, H, T, head_dim) so every head gets its own slice
        3. dot queries against keys, scaled by sqrt(head_dim) -> attn_scores
        4. mask the future positions to -inf, then softmax and drop out
        5. weight the values, concatenate the heads back to (B, T, C)
        6. mix the heads through o_proj -> returned attn_output
        """
        batch_size, seq_len, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        # Split the embedding across heads: (B, T, C) -> (B, H, T, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # How much each token cares about every other token, scaled by sqrt(head_dim).
        attn_scores = query_states @ key_states.transpose(-2, -1) / self.head_dim**0.5   # (B, H, T, T)

        # No peeking ahead: future positions become -inf, so softmax sends them to 0.
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len], float("-inf")
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)       # (B, H, T, T)
        attn_weights = self.dropout(attn_weights)

        # Weighted average of the values.
        attn_output = attn_weights @ value_states               # (B, H, T, head_dim)

        # Concatenate the heads back into one vector per token.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)


class GPTModel(nn.Module):
    """The full GPT-2 decoder: embeddings, a stack of pre-norm blocks, and an output head."""

    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.embed_positions = nn.Embedding(cfg["max_position_embeddings"], cfg["hidden_size"])

        self.layers = nn.ModuleList()
        for _ in range(cfg["num_hidden_layers"]):
            self.layers.append(nn.ModuleDict({
                # Layer norm before attention
                "input_layernorm": nn.LayerNorm(cfg["hidden_size"]),
                # Multi-head self-attention
                "self_attn": MultiheadAttention(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_attention_heads"],
                    max_position_embeddings=cfg["max_position_embeddings"],
                    attention_dropout=cfg["attention_dropout"],
                    attention_bias=cfg["attention_bias"],
                ),
                # Layer norm before MLP
                "post_attention_layernorm": nn.LayerNorm(cfg["hidden_size"]),
                # Feed-forward MLP: cfg["hidden_size"] -> 4x cfg["hidden_size"] -> cfg["hidden_size"]
                "mlp": nn.Sequential(
                    nn.Linear(cfg["hidden_size"], 4 * cfg["hidden_size"]),
                    nn.GELU(),
                    nn.Linear(4 * cfg["hidden_size"], cfg["hidden_size"]),
                ),
            }))

        self.norm = nn.LayerNorm(cfg["hidden_size"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids):
        """Turns a batch of token ids into next-token scores over the vocabulary.

        1. look up a token embedding and a position embedding, and add them
        2. for each layer: pre-norm, self-attend, add the residual
        3. for each layer: pre-norm, run the MLP, add the residual
        4. apply the final layer norm
        5. project to vocabulary size through lm_head -> returned logits
        """
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(torch.arange(seq_len, device=input_ids.device))
        x = inputs_embeds + position_embeds  # Shape [batch_size, seq_len, hidden_size]

        for layer in self.layers:
            # Pre-norm + self-attention + residual connection
            normed = layer["input_layernorm"](x)
            x = x + layer["self_attn"](normed)  # Residual connection

            # Pre-norm + MLP + residual connection
            normed = layer["post_attention_layernorm"](x)
            x = x + layer["mlp"](normed)  # Residual connection

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def generate(model, input_ids, max_new_tokens, max_seq_len):
    """Extends input_ids by max_new_tokens, greedily, without a KV cache.

    1. crop the sequence to the last max_seq_len tokens the model supports
    2. run the whole cropped sequence through the model -> logits
    3. keep only the last position, since that is what predicts the next token
    4. softmax and take the argmax -> next_token
    5. append it and repeat -> returned input_ids, prompt plus continuation

    Each step re-processes the full sequence. That is intentionally naive.
    """
    for _ in range(max_new_tokens):
        # Crop to the context the model supports. E.g. if the model supports
        # 5 tokens and we already have 10, only the last 5 are fed in.
        context_ids = input_ids[:, -max_seq_len:]

        logits = model(context_ids)          # (B, T, vocab_size)

        # Only the LAST position predicts the next token.
        next_token_logits = logits[:, -1, :]                     # (B, vocab_size)
        probs = torch.softmax(next_token_logits, dim=-1)         # (B, vocab_size)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)   # (B, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__} | Device: cpu")

    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hi my name is"
    input_ids = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)

    model = GPTModel(GPT_CONFIG_124M)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    model.eval()  # disable dropout

    out = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=6,
        max_seq_len=GPT_CONFIG_124M["max_position_embeddings"],
    )

    print(tokenizer.decode(out[0].tolist()))
