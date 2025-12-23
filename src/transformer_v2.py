import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_src_mask(src, pad_id):
    return (src != pad_id).unsqueeze(1).unsqueeze(2)
def make_tgt_mask(tgt, pad_id):
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    nopeak = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=tgt.device)).bool()
    return pad_mask & nopeak

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
        bsz, q_len, _ = x.size()
        k_len = context.size(1)
        q = self.q_proj(x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask):
        x = x + self.dropout(self.attn(self.norm1(x), mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=enc_out, mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

class TransformerV2(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.token_embed.weight
    def encode(self, src, src_mask):
        x = self.pos(self.token_embed(src))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x
    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos(self.token_embed(tgt))
        for layer in self.dec_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.out(x)
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        return self.decode(tgt, enc_out, src_mask, tgt_mask)
