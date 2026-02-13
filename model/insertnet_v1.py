''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class OutputAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        n_head = 1
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.temperature = d_k ** 0.5

    def forward(self, q, k, v, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        attn_before_softmax = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn_before_softmax = attn_before_softmax.masked_fill(mask == 0, -1e9)

        return attn_before_softmax

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, max_seq_len=256, dropout=0.1, scale_emb=False):

        super().__init__()

        self.position_enc = PositionalEncoding(512, n_position=max_seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        bias_table = np.zeros([max_seq_len, max_seq_len])
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                bias_table[i, j] = j - i
        bias_table = torch.from_numpy(np.array(bias_table)).float() / max_seq_len
        self.register_buffer('bias_table', bias_table)

    def forward(self, src_seq, src_mask, voxsz, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, enc_output1, enc_output2,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_enc_attn = self.enc_attn(
            enc_output2, enc_output1, enc_output1, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, scale_emb=False):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, enc_output1, enc_output2, mask1, return_attns=False):

        dec_enc_attn_list = []
        dec_output = enc_output2
        # -- Forward
        for dec_layer in self.layer_stack:
            dec_output, dec_enc_attn = dec_layer(
                enc_output1, dec_output, slf_attn_mask=None, dec_enc_attn_mask=mask1)
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_enc_attn_list
        return dec_output,

class InsertionNet(nn.Module):

    def __init__(self, im_tokenizer, n_enc_layers=4, n_dec_layers=4, 
                 d_model=512, d_inner=2048, n_head=8, d_k=64, d_v=64, max_seq_len=256):

        super().__init__()

        self.im_tokenizer = im_tokenizer

        self.encoder = Encoder(
            n_layers=n_enc_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_inner=d_inner,
            max_seq_len=max_seq_len
        )

        self.decoder = Decoder(
            n_layers=n_dec_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_inner=d_inner
        )

        self.output_attn = OutputAttention(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model
        )
    
    def forward(self, stack1, stack2, mask1, mask2, vox1, vox2):

        mask1.unsqueeze(1).repeat(1, mask1.shape[1], 1).shape
        mask2.unsqueeze(1).repeat(1, mask2.shape[1], 1).shape

        feat1 = self.im_tokenizer(stack1)[None]
        feat2 = self.im_tokenizer(stack2)[None]

        self_feat1 = self.encoder(feat1, mask1, vox1)[0]
        self_feat2 = self.encoder(feat2, mask2, vox2)[0]

        cross_feat2 = self.decoder(self_feat1, self_feat2, mask1)[0]
        attn_2to1 = self.output_attn(q=cross_feat2, k=self_feat1, v=self_feat1, mask=mask1)

        return attn_2to1

if __name__ == '__main__':
    import torchvision

    im_tokenizer = torchvision.models.resnet18()
    im_tokenizer.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    im_tokenizer.fc = torch.nn.Identity()

    model = InsertionNet(
        im_tokenizer=im_tokenizer,
        d_model=512,
        max_seq_len=200
    )

    device = 'cuda'

    stack1 = torch.rand((200, 1, 256, 256)).to(device)
    stack2 = torch.rand((200, 1, 256, 256)).to(device)
    mask1 = torch.zeros(200).to(device)
    mask1[: 150] = 1
    mask2 = torch.zeros(200).to(device)
    mask1[: 130] = 1
    vox1 = torch.rand(3).to(device)
    vox2 = torch.rand(3).to(device)

    model.to(device)

    y = model(stack1, stack2, mask1[None], mask2[None], vox1, vox2)