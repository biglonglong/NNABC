import torch
from torch import nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)

    def forward(self, x):
        # [batch_size, seq_len]
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                     (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [batch_size, seq_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(self.head_dim)
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: (seq_len_q, seq_len_k) or (batch_size, n_heads, seq_len_q, seq_len_k)
            key_padding_mask: (batch_size, seq_len_k)
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attn_weights: (batch_size, n_heads, seq_len_q, seq_len_k)
        """

        batch_size = query.size(0)

        # [batch_size, n_heads, seq_len_q, head_dim]
        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # [batch_size, n_heads, seq_len_k, head_dim]
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # [batch_size, n_heads, seq_len_v, head_dim]
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # [batch_size, n_heads, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            if mask.dim() == 2:
                # [1, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        if key_padding_mask is not None:
            # [batch_size, 1, 1, seq_len_k]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        attn_weights = attn_scores.softmax(dim=-1)
        dropped_attn_weights = self.dropout(attn_weights)

        # [batch_size, n_heads, seq_len_q, head_dim]
        attn_output = torch.matmul(dropped_attn_weights, v)
        # [batch_size, seq_len_q, d_model]
        multi_attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(multi_attn_output)

        return output, attn_weights


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, d_model),
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Q=K=V=Embeddings
        attn_output, _ = self.self_attn(
            src, src, src,
            mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout1(attn_output))

        src2 = self.ffn(src)
        src = self.norm2(src + self.dropout2(src2))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, d_model),
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Q=K=V=Embeddings
        self_attn_output, _ = self.self_attn(
            tgt, tgt, tgt,
            mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = self.norm1(tgt + self.dropout1(self_attn_output))

        if memory is None:
            cross_attn_output, _ = self.cross_attn(
                tgt, memory, memory,
                mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            tgt = self.norm2(tgt + self.dropout2(cross_attn_output))

        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ff_hidden_dim=2048, 
                 dropout_rate=0.1, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ff_hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ff_hidden_dim=2048, 
                 dropout_rate=0.1, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, ff_hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return tgt


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, ff_hidden_dim=2048, 
                 max_len=5000, dropout_rate=0.1, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_len, dropout_rate)

        self.encoder = TransformerEncoder(
            d_model, n_heads, ff_hidden_dim, dropout_rate, num_encoder_layers
        )
        self.decoder = TransformerDecoder(
            d_model, n_heads, ff_hidden_dim, dropout_rate, num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        src_emb = self.src_embedding(src)
        src_emb = self.positional_embedding(src_emb)
        memory = self.encoder(src_emb, src_mask, src_key_padding_mask)

        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.positional_embedding(tgt_emb)
 
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        output = self.fc_out(decoder_output)
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 32
    src_seq_len = 20
    tgt_seq_len = 15
    src_vocab_size = 10000
    tgt_vocab_size = 8000

    model = Transformer(src_vocab_size, tgt_vocab_size, 512, 8, 2048, 5000, 0.1, 6, 6).to(device)

    # input shape: [batch_size, seq_len, embedding_dim]
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    
    # different MultiHeadAttention's mask is different [q, k]
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool().to(device)
    
    padding_index = 1
    tgt_key_padding_mask = (tgt == padding_index)

    output = model(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    print("output.shape:", output.shape)


