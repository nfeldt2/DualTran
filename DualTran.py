import torch
import torch.nn as nn
import math
from einops import rearrange


class DualTran(nn.Module):
    def __init__(self, num_classes=10, emb_size=32, device='cuda'):
        super(DualTran, self).__init__()
        self.channel_tran = ChannelTran().to(device)
        self.time_tran = TimeTran(emb_size=emb_size).to(device)

    def forward(self, x):
        x = self.channel_tran(x)
        x = self.time_tran(x)
        return x


class TimeTran(nn.Module):
    def __init__(self, in_channels=32, d_model=350, emb_size=32, dim_ff=512):
        super(TimeTran, self).__init__()
        self.embedding = TimeEmbedding(emb_size=32, channel_size=d_model)
        self.pos_enc = AbsPosEnc(d_model=emb_size, dropout=0.1, max_len=350)
        self.time_attn = MultiHeadAttention(d_model=emb_size, num_heads=8)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(0.15))
        
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size*d_model, 10)

    def forward(self, x):
        x_src_pos = self.pos_enc(x)
        att = x + self.time_attn(x_src_pos)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.flatten(out)
        out = self.out(out)
        return out  


class ChannelTran(nn.Module):
    def __init__(self, in_channels=128, out_channels=32, d_model=400):
        super(ChannelTran, self).__init__()
        self.embedding = ChannelEmbedding(in_channels, seq_len=500)
        self.learned_pos = LearnedPosEnc(in_channels, d_model)
        self.channel_attn = MultiHeadAttention(d_model, num_heads=10)
        self.fcn_1 = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                    nn.GELU(),
                                    nn.Linear(d_model * 2, d_model - 50)
                                    )
        self.fcn_2 = nn.Sequential(nn.Linear(in_channels, in_channels * 4),
                                    nn.GELU(),
                                    nn.Linear(in_channels * 4, in_channels - 32),
                                    nn.GELU(),
                                    nn.Linear(in_channels - 32, (in_channels - 32) * 4), 
                                    nn.GELU(),
                                    nn.Linear((in_channels - 32) * 4, in_channels - 64),
                                    nn.GELU(),
                                    nn.Linear(in_channels - 64, out_channels)
                                    )
        
        self.layer_norm1 = nn.LayerNorm(d_model-50, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(out_channels, eps=1e-5)
        
    def forward(self, x):
        x_emb = self.embedding(x)
        x_pos = self.learned_pos(x_emb)
        x_attn = self.channel_attn(x_pos)
        x_fcn = self.fcn_1(x_attn)
        x_fcn = self.layer_norm1(x_fcn)
        x_fcn = x_fcn.permute(0, 2, 1)
        x_fcn = self.fcn_2(x_fcn)
        x_fcn = self.layer_norm2(x_fcn)
        return x_fcn
        

class LearnedPosEnc(nn.Module):
    def __init__(self, num_positions, d_model):
        super(LearnedPosEnc, self).__init__()
        self.pos_enc = nn.Embedding(num_positions, d_model)
        nn.init.normal_(self.pos_enc.weight, mean=0, std=d_model**-0.5)
        self.positions = torch.arange(0, num_positions, dtype=torch.long).unsqueeze(0)

    def forward(self, x):
        positions = self.positions.expand(x.size(0), -1).to(x.device)
        pos_enc = self.pos_enc(positions)
        return x + pos_enc


class ChannelEmbedding(nn.Module):
    def __init__(self, in_channels, seq_len=500):
        super(ChannelEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels * 4, kernel_size=3, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels * 4)
        self.fc1 = nn.Linear(seq_len*4, seq_len - 50)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels, in_channels * 4, kernel_size=3, padding=1, groups=in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels * 4)
        self.fc2 = nn.Linear((seq_len-50)*4, seq_len - 100)
        self.in_channels = in_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x_flat = x.view(x.size(0), self.in_channels, -1)
        x = self.fc1(x_flat)
        

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        x_flat = x.view(x.size(0), self.in_channels, -1)
        x = self.fc2(x_flat)

        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "Embedding size must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.scaling = self.depth ** -0.5

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)

        attention = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scaling
        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3)

        # why is data size (batch, 128, 10, 40)? 

        out = out.contiguous().view(batch_size, -1, self.d_model)
        out = self.out(out)

        return out
    

class TimeEmbedding(nn.Module):
    def __init__(self, emb_size=32, channel_size=350, channels=32):
        super(TimeEmbedding, self).__init__()

        self.embed_layer1 = nn.Sequential(nn.Conv1d(channels, channels * 4, kernel_size=3, padding=1, groups=channels),
                                                nn.BatchNorm1d(channels * 4),
                                                nn.GELU(), 
                                                nn.Conv1d(channels * 4, emb_size, kernel_size=1, groups=channels),
                                                nn.BatchNorm1d(emb_size),
                                                nn.GELU())
        
        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size, emb_size * 4, kernel_size=[channel_size//4, 1], padding=[channel_size//8, 0]),
                                                nn.BatchNorm2d(emb_size * 4),
                                                nn.GELU(),
                                                nn.Conv2d(emb_size * 4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                                nn.BatchNorm2d(emb_size),
                                                nn.GELU())
    
    def forward(self, x):
        x = self.embed_layer1(x)
        x = self.embed_layer2(x.unsqueeze(1)).squeeze(2)
        x = x.permute(0, 2, 1)
        return x
    

class AbsPosEnc(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model=350, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsPosEnc, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)
    

class AttentionWithRelPosEnc(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out
        