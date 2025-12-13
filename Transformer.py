import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

torch.manual_seed(42)
np.random.seed(42)


class RMS_Norm(nn.Module):
    def __init__(self, embed_dim: int, epsilon: float = 1e-6):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(embed_dim))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        """
        args :

            hidden_states : input tensor of shape (batch_size, sequence_len, embed_dim)

            initial values of x = [3, 4, 5]

            # step 1: square each value
            x² = [9, 16, 25]

            # step 2: calculate mean
            mean = (9 + 16 + 25) / 3 = 50/3 ≈ 16.67

            # step 3: take square root
            rms = sqrt(16.67) ≈ 4.08

            # step 4: normalize (divide by rms)
            normalized = [3/4.08, 4/4.08, 5/4.08]
            normalized ≈ [0.735, 0.980, 1.225]

        returns :

            normalized tensor of same shape

        """
        # calculating rms for each position
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states / torch.sqrt(variance + self.epsilon)

        return hidden_states * self.weights


class Rotary_PositionalEmebedding(nn.Module):
    """
    rotary position embedding (rope)

    imagine spinning a wheel - each position on the wheel (token position)
    gets a unique rotation. this helps the model understand "what comes before what".

    """

    def __init__(
        self, head_dim: int, max_seq_length: int = 2048, base: float = 10000.0
    ):
        """
        args:
            head_dim: dimension of each attention head (must be even)
            max_seq_length: maximum sequence length we'll handle
            base: base for the frequency calculation (10000 is standard)
        """
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        self.base = base

        # pre-compute the rotation frequencies for efficiency
        # these determine how fast each dimension rotates
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # pre-compute cos and sin values for all positions
        self._set_cos_sin_cache(max_seq_length)

    def _set_cos_sin_cache(self, seq_length: int):
        """pre-compute cosine and sine values for all positions"""
        # create position indices [0, 1, 2, ..., seq_length-1]
        position = torch.arange(seq_length).float()

        # calculate frequencies for each position
        # outer product: each position × each frequency
        freqs = torch.outer(position, self.inv_freq)

        # duplicate frequencies (for pairs of dimensions)
        emb = torch.cat([freqs, freqs], dim=-1)

        # pre-compute cos and sin for efficiency
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x):
        """
        rotate half the hidden dims of the input.
        this is the core rope operation.

        example: [a, b, c, d] -> [-c, -d, a, b]
        """
        # split the last dimension in half
        x1 = x[..., : x.shape[-1] // 2]  # first half
        x2 = x[..., x.shape[-1] // 2 :]  # second half

        # rotate: swap and negate
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, query, key, position_ids=None):
        """
        apply rotary embeddings to query and key tensors

        args:
            query: query tensor (batch, num_heads, seq_len, head_dim)
            key: key tensor (batch, num_heads, seq_len, head_dim)
            position_ids: optional specific positions (usually just 0,1,2,...)

        returns:
            rotated query and key tensors
        """
        seq_len = query.shape[2]

        # get cos and sin values for these positions
        if position_ids is None:
            # use default sequential positions
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]

        # add dimensions for broadcasting
        # shape: (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # apply rotation using the formula:
        # x_rotated = x * cos + rotate_half(x) * sin
        query_rotated = (query * cos) + (self.rotate_half(query) * sin)
        key_rotated = (key * cos) + (self.rotate_half(key) * sin)

        return query_rotated, key_rotated


class Multi_Head_SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout_rate: float = 0.12,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len

        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # query, key, and value projection
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # attention dropout and output dropout
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

        self.rope = Rotary_PositionalEmebedding(
            head_dim=self.head_dim, max_seq_length=max_seq_len
        )

    def split_heads(self, tensor, batch_size):
        """

        splitting the last dimension into -> (num_heads, head_dim)

        transforms from: (batch, seq_len, embed_dim)
        to: (batch, num_heads, seq_len, head_dim)

        """

        # reshape to (batch_size, seq_len, num_heads, head_dim)
        tensor = tensor.view(batch_size, -1, self.num_heads, self.head_dim)

        # transpose to (batch_size, num_heads, seq_len, head_dim)
        # this puts heads in the second dimension for parallel processing

        return tensor.transpose(1, 2)

    def combine_heads(self, tensor, batch_size):
        """

        combine heads back to single hidden dimension

        transforms from: (batch, num_heads, seq_len, head_dim)
        to: (batch, seq_len, embed_dim)

        """

        # transpose to (batch, seq_len, num_heads, head_dim)
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, -1, self.embed_dim)

    def forward(self, hidden_state, attention_mask=None):
        """
        apply multi-head self-attention

        args:
            hidden_states: input tensor (batch, seq_len, embed_dim)
            attention_mask: optional mask to prevent attending to certain positions

        returns:
            output tensor (batch, seq_len, embed_dim)
        """

        batch_size = hidden_state.shape[0]

        # step 1 -> project inputs to query, key, value
        query = self.query_proj(hidden_state)  # (batch, seq_len, hidden)
        key = self.key_proj(hidden_state)  # (batch, seq_len, hidden)
        value = self.value_proj(hidden_state)  # (batch, seq_len, hidden)

        # step 2
        # split into multiple heads (batch, heads, seq_len, head_dim)
        query_layer = self.split_heads(query, batch_size=batch_size)
        key_layer = self.split_heads(key, batch_size=batch_size)
        value_layer = self.split_heads(value, batch_size=batch_size)

        # step 3: apply rotary position embeddings to q and k
        # this encodes position information directly into the representations
        query_layer, key_layer = self.rope(query_layer, key_layer)

        # step 4: calculate attention scores
        # Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        # QK^T: (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
        # result: (batch, heads, seq_len, seq_len) - attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        # scale down attention scores (prevents softmax saturation)
        attention_scores = attention_scores / self.scale

        # step 5: apply attention mask if provided
        # mask out padding tokens or future tokens (in decoder)
        if attention_mask is not None:
            # add very negative number to masked positions
            # after softmax, these become ~0
            attention_scores = attention_scores + attention_mask

        # convert the scores to probabilities using softmax
        attention_probs = f.softmax(attention_scores, dim=-1)

        # step 7: apply attention to values
        # multiply attention probabilities by values to get weighted sum
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim)
        # result: (batch, heads, seq_len, head_dim)
        context_layer = torch.matmul(attention_probs, value_layer)

        # step 8: combine all heads back together
        context_layer = self.combine_heads(context_layer, batch_size)

        # step 9: final Linear projection
        output = self.output_proj(context_layer)

        # apply dropout before returning
        output = self.output_dropout(output)

        return output


class SwiGLU_Feed_Forward(nn.Module):
    def __init__(
        self, embed_dim: int, intermediate_dim: int, dropout_rate: float = 0.1
    ):
        """
        Args :
            embed_dim = embed_dim
            intermediate_dim = 4*embed_dim
            dropout_rate = dropout_probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate

        # First step is expansion : projection to intermediate state
        # This creates richer representation with more capacity.
        self.gate_projection = nn.Linear(embed_dim, intermediate_dim, bias=False)

        # Second expansion : parallel path for gating
        # Decides what information to let through.
        self.up_projection = nn.Linear(embed_dim, intermediate_dim, bias=False)

        # Down projection : compress back to embed_dim
        # Integrates all the processed information.
        self.down_projection = nn.Linear(intermediate_dim, embed_dim, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_state):
        """
        Applying SwiGLU Feed Forward transformation
        Args :
            hidden_state : (batch_size, seq_len, embed_dim)
        Returns :
            output : (batch_size, seq_len, embed_dim)
        """

        # Expanding to intermediate size of applying silu activation (x * sigmoid(x))
        gate = f.silu(self.gate_projection(hidden_state))
        up = self.up_projection(hidden_state)

        # Step 3: Element-wise multiplication (gating)
        # The gate controls how much of 'up' passes through
        # This is the GLU (Gated Linear Unit) part
        gated = gate * up

        # Projecting back down to embed_dim
        down = self.down_projection(gated)
        output = self.dropout(down)

        return output


class TransformerBlock(nn.Module):
    """
    A single transformer block layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_dim: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            embed_dim : (512, 768)
            num_heads : embed_dim % num_heads = 0 | (2, 4)
            intermediate_dim : 4x embed_dim | (2048)
            dropout_rate : dropout_probability
        Returns:

        """

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len

        # Pre-norm initialization
        # self.attention_rms_norm -> before calculating attention
        self.attention_rms_norm = RMS_Norm(embed_dim=embed_dim)
        # self.SwiGLU_ffn_rms_norm -> before SwiGLU_ffn
        self.SwiGLU_ffn_rms_norm = RMS_Norm(embed_dim=embed_dim)

        # Attention layer
        self.mhsa = Multi_Head_SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
        )
        # SwiGLU_ffn layer
        self.SwiGLU_ffn = SwiGLU_Feed_Forward(
            embed_dim=embed_dim,
            intermediate_dim=intermediate_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, hidden_state, attention_mask=None):
        """
        Args:
            hidden_state : input tensor (batch_size, seq_len, embed_dim),
            attention_mask : optional attention mask
        Returns:
            hidden_state : output tensor (batch_size, seq_len, embed_dim)
        """

        # Saving the current hidden state for residual connection
        residual = hidden_state
        # Applying pre-normalization
        hidden_state = self.attention_rms_norm(hidden_state)
        # Applying attention
        hidden_state = self.mhsa(hidden_state, attention_mask)
        # Applying residual connection
        hidden_state = hidden_state + residual

        # Saving the current state for residual connection
        residual = hidden_state
        # Applying pre-normalization
        hidden_state = self.SwiGLU_ffn_rms_norm(hidden_state)
        # Feeding through SwiGLU_ffn
        hidden_state = self.SwiGLU_ffn(hidden_state)
        # Applying residual connection
        hidden_state = hidden_state + residual

        return hidden_state


class Transformer(nn.Module):
    """
    A complete Transformer model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        intermediate_dim: int,
        num_layers: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id

        # Converting token ids into dense vectors
        # Each token gets a learnable vector representation
        self.token_embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=self.pad_token_id,
        )

        # Embedding dropout for regularization
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # Transformer blocks (main processing layer)
        # stacking multiple transformer blocks
        # more blocks -> deeper model -> more capacity to learn complex pattern
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    max_seq_len=max_seq_len,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        # Final Norm (normalize the output before final prediction)
        self.final_norm = RMS_Norm(embed_dim=embed_dim)

        # Output Head
        # Projecting hidden states back to vocabulary for prediction
        self.output_head = nn.Linear(
            in_features=embed_dim, out_features=self.vocab_size, bias=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Xavier/Glorot initialization for linear layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                # Ensuring padding token has zero embedding
                module.weight.data[module.padding_idx].zero_()

    def create_attention_mask(self, input_ids):
        """
        Create attention mask to ignore padding tokens

        Args:
            input_ids: Token IDs (batch, seq_len)

        Returns:
            Attention mask (batch, 1, 1, seq_len)
        """
        # Create mask: 1 for real tokens, 0 for padding
        mask = (input_ids != self.pad_token_id).float()

        # Reshape for broadcasting: (batch, 1, 1, seq_len)
        mask = mask.unsqueeze(1).unsqueeze(2)

        # Convert to attention scores format
        # 0 for attend, large negative for ignore
        attention_mask = (1.0 - mask) * -10000.0

        return attention_mask

    def forward(self, input_ids, attention_mask=None, return_hidden_states=False):
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids=input_ids)
        # input ids -> (batch_size, seq_len)
        hidden_state = self.token_embedding(
            input_ids
        )  # hidden_state -> (batch_size, seq_len, embed_dim)

        # Applying embedding dropout
        hidden_state = self.embedding_dropout(hidden_state)

        # Storing the hidden states
        all_hidden_states = [] if return_hidden_states else None
        # Passing through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if return_hidden_states:
                all_hidden_states.append(hidden_state)
            # Applying transformer block
            hidden_state = block(hidden_state, attention_mask)

        # Applying final normalization
        hidden_state = self.final_norm(hidden_state)

        if return_hidden_states:
            all_hidden_states.append(hidden_state)

        logits = self.output_head(hidden_state)

        if return_hidden_states:
            return logits, all_hidden_states
        return logits

    def generate(
        self,
        input_ids,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k=None,
    ):
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # if input_ids grows beyong max_seq_len, rope will crash
                # Therefore trunctation is needed.
                if input_ids.shape[1] > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len :]

                logits = self.forward(input_ids=input_ids)
                logits = logits[:, -1, :]
                logits = logits / temperature

                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float("-inf"))
                    logits = logits.scatter_(
                        dim=1, index=top_k_indices, src=top_k_logits
                    )

                # Converting to probabilities
                probs = f.softmax(logits, dim=-1)

                # Sampling next token
                next_token = torch.multinomial(input=probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                if (next_token == self.tokenizer.eos_token_id).all():
                    break

        return input_ids
