import torch
import torch.nn as nn
from modules import *

class TSRNet(nn.Module):

    def __init__(self, enc_in):
        super(TSRNet, self).__init__()

        self.channel = enc_in

        # BCG 신호를 위한 시계열 모듈
        self.time_encoder = Encoder1D(enc_in)
        self.time_decoder = Decoder1D(enc_in + 1)
        
        # BCG 신호를 위한 스펙트로그램 모듈
        self.spec_encoder = Encoder2D(enc_in)
        # 인코딩된 스펙트로그램 특징을 처리하는 레이어
        self.conv_spec1 = nn.Conv1d(50 * 51, 50, 3, 1, 1, bias=False)
        
        # 결합된 특징을 처리하는 레이어
        self.mlp = nn.Sequential(
            nn.Linear(202, 136),
            nn.LayerNorm(136),
            nn.ReLU()
        )
        
        # 결합된 특징을 위한 Attention 레이어
        self.attn1 = MultiHeadedAttention(2, 50)
        self.drop = nn.Dropout(0.1)
        self.layer_norm1 = LayerNorm(50)

    def attention_func(self, x, attn, norm):
        attn_latent = attn(x, x, x)
        attn_latent = norm(x + self.drop(attn_latent))
        return attn_latent
    
    def forward(self, time_bcg, spectrogram_bcg):
        # BCG 시계열 데이터 인코딩
        time_features = self.time_encoder(time_bcg.transpose(-1, 1))  # (batch_size, channels, length)

        # BCG 스펙트로그램 인코딩
        spectrogram_features = self.spec_encoder(spectrogram_bcg.permute(0, 3, 1, 2))  # (batch_size, channels, height, width)
        n, c, h, w = spectrogram_features.shape
        spectrogram_features = self.conv_spec1(spectrogram_features.contiguous().view(n, c * h, w))  # (batch_size, 50, width)

        # 시계열 데이터와 스펙트로그램 특징 결합
        latent_combine = torch.cat([time_features, spectrogram_features], dim=-1)

        # Cross-attention 메커니즘
        latent_combine = latent_combine.transpose(-1, 1)
        attn_latent = self.attention_func(latent_combine, self.attn1, self.layer_norm1)
        attn_latent = self.attention_func(attn_latent, self.attn1, self.layer_norm1)
        latent_combine = attn_latent.transpose(-1, 1)

        # MLP 레이어를 통해 처리
        latent_combine = self.mlp(latent_combine)

        # 인코딩된 특징을 다시 원래 시계열 신호로 디코딩
        output = self.time_decoder(latent_combine)
        output = output.transpose(-1, 1)

        return output[:, :, 0:self.channel], output[:, :, self.channel:self.channel + 1]
