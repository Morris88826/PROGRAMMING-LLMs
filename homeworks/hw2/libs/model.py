import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, PretrainedConfig, PreTrainedModel
try:
    from libs.utils import print0, remove_prefix_from_state_dict
except ImportError:
    from utils import print0, remove_prefix_from_state_dict

class GPTConfig(PretrainedConfig):
    model_type = "custom-gpt2"
    def __init__(self, block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, norm_method="layernorm", act_method="gelu", RoPE=False, group_size=1, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size # maximum sequence length
        self.vocab_size = vocab_size # number of tokens
        self.n_layer = n_layer # number of layers
        self.n_head = n_head  # number of heads, d_head = d_model // n_head
        self.n_embd = n_embd # d_model

        ### Additional parameters
        self.norm_method = norm_method # layernorm or rmsnorm
        self.act_method = act_method # gelu or swiglu
        self.RoPE = RoPE # use Rotary Positional Embeddings
        self.group_size = group_size # group size for Grouped Query Attention


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float() # outer product: ix1 * 1xj
        
        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

        # cache[m, k, 0] = cos(m * theta[k])
        # cache[m, k, 1] = sin(m * theta[k])

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2) # expand the batch and head dimensions

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

# multi-head attention (causal self-attention for autoregressive models)
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, FLASH=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.group_size == 0

        self.group_size = config.group_size

        # query, key, value projection for all heads
        self.wq = nn.Linear(config.n_embd, config.n_embd)
        self.wk = nn.Linear(config.n_embd, config.n_embd // self.group_size)
        self.wv = nn.Linear(config.n_embd, config.n_embd // self.group_size)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.use_FLASH = FLASH
        self.use_RoPE = config.RoPE
        if self.use_RoPE:
            self.RoPE = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=config.block_size * 2) # calculate a longer sequence length for RoPE


        # bias for masked attention (lower triangular matrix), register as buffer (not learnable). Set the block size to the maximum sequence length so that we can reuse the same bias for all sequence lengths.
        # don't change the upper triangular part to -inf here for efficiency, we will do it on the fly during the attention operation (only change the values that we will actually, i.e. the sequence length)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, n_embd
        assert C == self.n_embd

        q = self.wq(x) # B, T, n_embd 
        k = self.wk(x) # B, T, n_embd // n_head
        v = self.wv(x) # B, T, n_embd // n_head

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B, n_head, T, head_dim
        k = k.view(B, T, self.n_head // self.group_size, self.head_dim).transpose(1, 2) # B, n_head/group_size, T, head_dim
        v = v.view(B, T, self.n_head // self.group_size, self.head_dim).transpose(1, 2) # B, n_head/group_size, T, head_dim

        assert q.size() == (B, self.n_head, T, self.head_dim)
        assert k.size() == (B, self.n_head // self.group_size, T, self.head_dim)
        assert v.size() == (B, self.n_head // self.group_size, T, self.head_dim)

        # attention
        if self.use_RoPE:
            q = self.RoPE(q)
            k = self.RoPE(k)

        if self.group_size > 1: # repeat k and v for each group
            k = k[:, :, None, :, :].expand(B, self.n_head//self.group_size, self.group_size, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_head//self.group_size, self.group_size, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
    
        if self.use_FLASH:
            # optimized version of the attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # mask out the upper triangular part so that we attend only to the left in the input sequence
            att = F.softmax(att, dim=-1)
            y = torch.matmul(att, v) # B, n_head, T, hs

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd) # B, T, n_embd
        y = self.c_proj(y)
        return y

class FFN_SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2/3 * 4 * config.n_embd) # suggested by GLU Variants paper
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.v = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.act_layer = nn.SiLU()
    
    def forward(self, x):
        swish = self.act_layer(self.w1(x))
        x_V = self.v(x)
        x = swish * x_V
        x = self.w2(x)
        return x
    

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act_layer = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act_layer(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config: GPTConfig, FLASH=False):
        super().__init__()

        norm_layer = nn.LayerNorm
        if config.norm_method == "rmsnorm":
            norm_layer = nn.RMSNorm
        
        self.n_1 = norm_layer(config.n_embd)
        self.attn = CausalSelfAttention(config, FLASH=FLASH)
        self.n_2 = norm_layer(config.n_embd)
        if config.act_method == "gelu":
            self.mlp = FFN(config)
        elif config.act_method == "swiglu":
            self.mlp = FFN_SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.n_1(x))
        x = x + self.mlp(self.n_2(x))
        return x


# norm_method="layernorm", act_method="gelu", use_RoPE=False, use_FLASH=False, group_size=1
class GPT(PreTrainedModel):
    config_class = GPTConfig
    def __init__(self, config: GPTConfig, use_FLASH=False):
        super().__init__(config)
        self.config = config
        self.FLASH = use_FLASH

        norm_layer = nn.LayerNorm
        if config.norm_method == "rmsnorm":
            norm_layer = nn.RMSNorm

        if config.RoPE:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                h = nn.ModuleList([Block(config, FLASH=self.FLASH) for _ in range(config.n_layer)]),
                ln_f = norm_layer(config.n_embd),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config, FLASH=self.FLASH) for _ in range(config.n_layer)]),
                ln_f = norm_layer(config.n_embd),
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 
        # weight sharing
        self.transformer["wte"].weight = self.lm_head.weight # copy by reference

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.95), eps=1e-8):
        param_dict = {k: p for k, p in self.named_parameters() }
        param_dict = {k: p for k, p in param_dict.items() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in no_decay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
        return optimizer
        

    def forward(self, idx, targets=None):
        # idx: B, T
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length is too long ({T} > {self.config.block_size})"

        pos = torch.arange(T, dtype=torch.long, device=idx.device) # T
        if self.config.RoPE:
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            x = tok_emb
        else: # use the original GPT-2 model: token embeddings + global position embeddings
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = tok_emb + pos_emb

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x) # B, T, n_embd
        logits = self.lm_head(x) # B, T, vocab_size

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            return logits

    @classmethod
    def from_pretrained_gpt(cls, model_type):
        """
        Load a pre-trained model from Hugging Face's transformers library.
        """
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        
        config_args = {
            "gpt2": GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50257), # 124M parameters
            "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024, vocab_size=50257), # 345M parameters
            "gpt2-large": GPTConfig(n_layer=36, n_head=20, n_embd=1280, block_size=1024, vocab_size=50257), # 774M parameters
            "gpt2-xl": GPTConfig(n_layer=48, n_head=25, n_embd=1600, block_size=1024, vocab_size=50257) # 1558M parameters
        }

        config = config_args[model_type]
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")] # remove the bias for masked attention

        # load the pre-trained model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy the weights 
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")] # remove the bias for masked attention
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"{len(sd_keys)} != {len(sd_keys_hf)}"

        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

if __name__ == "__main__":
    config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    model = GPT(config)
    print("Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")

    