import random

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, hidden, cell, teacher_forcing_ratio=0.5):
        target_len, batch_size = target.shape[0], target.shape[1]
        target_vocab_size = self.one_step_decoder.input_output_dim
        # Store the predictions in an array for loss calculations
        predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        # Take the very first word token, which will be sos
        input = target[0, :]

        # Loop through all the time steps, starts from 1
        for t in range(1, target_len):
            predict, hidden, cell = self.one_step_decoder(input, hidden, cell)

            predictions[t] = predict
            input = predict.argmax(1)

            # Teacher forcing
            do_teacher_forcing = random.random() < teacher_forcing_ratio
            input = target[t] if do_teacher_forcing else input

        return predictions
        