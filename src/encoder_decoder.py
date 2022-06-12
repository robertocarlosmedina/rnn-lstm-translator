import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(source)
        outputs = self.decoder(target, hidden, cell, teacher_forcing_ratio)

        return outputs
