import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torch.utils.tensorboard import SummaryWriter
from nltk.tokenize.treebank import TreebankWordDetokenizer

import spacy
import numpy as np
import random
from tqdm import tqdm
from termcolor import colored
import random
import math
import os

from src.decoder import Decoder
from src.encoder import Encoder
from src.encoder_decoder import EncoderDecoder
from src.one_step_decoder import OneStepDecoder
from src.grammar_checker import Grammar_checker


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CLIP = 1
EPOCHS = 1

class Seq2Seq_Translator:

    spacy_cv = spacy.load('pt_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def __init__(self) -> None:
        self.get_datasets()
        self.create_model()
        self.grammar = Grammar_checker()
        self.writer = SummaryWriter()
        pass

    def tokenize_cv(self, text):
        return [token.text for token in self.spacy_cv.tokenizer(text)]
    
    def tokenize_en(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]

    def get_datasets(self, batch_size=128):
        # Download the language files
        
        # Create the pytext's Field
        self.source = Field(tokenize=self.tokenize_cv, init_token='<sos>', eos_token='<eos>', lower=True)
        self.target = Field(tokenize=self.tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

        # Splits the data in Train, Test and Validation data
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=(".cv", ".en"), fields=(self.source, self.target),
            test="test", path=".data/criolSet"
        )

        # Build the vocabulary for both the language
        self.source.build_vocab(self.train_data, min_freq=3)
        self.target.build_vocab(self.train_data, min_freq=3)

        # Create the Iterator using builtin Bucketing
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits((self.train_data, self.valid_data, self.test_data),
                                                                              batch_size=batch_size,
                                                                              sort_within_batch=True,
                                                                              sort_key=lambda x: len(x.src),
                                                                              device=device)

    def create_model(self):
        # Define the required dimensions and hyper parameters
        embedding_dim = 256
        hidden_dim = 1024
        dropout = 0.5

        # Instanciate the models
        encoder = Encoder(len(self.source.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
        one_step_decoder = OneStepDecoder(len(self.target.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
        decoder = Decoder(one_step_decoder, device)

        self.model = EncoderDecoder(encoder, decoder)

        self.model = self.model.to(device)

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters())

        # Makes sure the CrossEntropyLoss ignores the padding tokens.
        TARGET_PAD_IDX = self.target.vocab.stoi[self.target.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

        self.load_models()

    def load_models(self):
        print("=> Loading checkpoint")
        try:
            checkpoint = torch.load('checkpoints/nmt.model.lstm.25.pth.tar')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            print(colored("=> No checkpoint to Load", "red"))
    
    def save_model(self):
        print("=> Saving checkpoint")
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, 'checkpoints/nmt.model.lstm.25.pth.tar')
    
    def show_train_metrics(self, epoch: int, train_loss: float, 
        train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        print(f' Epoch: {epoch:03}/{EPOCHS}')
        print(
            f' Train Loss: {train_loss:.3f} | Train Acc: {train_accuracy:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f' Val. Loss: {valid_loss:.3f} | Val Acc: {valid_accuracy:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}\n')
    
    def save_train_metrics(self, epoch: int, train_loss: float, 
            train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        """
            Save the training metrics to be ploted in the tensorboard.
        """
        # All stand alone metrics
        self.writer.add_scalar(
            "Training Loss", train_loss, global_step=epoch)
        self.writer.add_scalar(
            "Training Accuracy", train_accuracy, global_step=epoch)
        self.writer.add_scalar(
            "Validation Loss", valid_loss, global_step=epoch)
        self.writer.add_scalar(
            "Validation Accuracy", valid_accuracy, global_step=epoch)
        
        # Mixing Train Metrics
        self.writer.add_scalars(
            "Training Metrics (Train Loss / Train Accurary)", {
                "Train Loss": train_loss, "Train Accurary": train_accuracy},
            global_step=epoch
        )

        # Mixing Validation Metrics
        self.writer.add_scalars(
            "Training Metrics (Validation Loss / Validation Accurary)", {
                "Validation Loss": valid_loss, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )
        
        # Mixing Train and Validation Metrics
        self.writer.add_scalars(
            "Training Metrics (Train Loss / Validation Loss)", {
                "Train Loss": train_loss, "Validation Loss": valid_loss},
            global_step=epoch
        )
        self.writer.add_scalars(
            "Training Metrics (Train Accurary / Validation Accuracy)", {
                "Train Accurary": train_accuracy, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )
        
    def train(self, epoch, progress_bar):
        
        target_count, correct_train = 0, 0
        training_loss = []
        training_accu = []
        # set training mode
        self.model.train()
        # Loop through the training batch
        for i, batch in enumerate(self.train_iterator):
            # Get the source and target tokens
            src = batch.src
            trg = batch.trg
            self.optimizer.zero_grad()
            # Forward pass
            output = self.model(src, trg)
            # reshape the output
            output_dim = output.shape[-1]
            # Discard the first token as this will always be 0
            output = output[1:].view(-1, output_dim)
            # Discard the sos token from target
            trg = trg[1:].view(-1)
            # Calculate the loss
            loss = self.criterion(output, trg)
            # back propagation
            loss.backward()
            # Gradient Clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
            self.optimizer.step()
            training_loss.append(loss.item())
            # Calculate Accuracy
            _, predicted = torch.max(output.data, 1)
            target_count += trg.size(0)
            correct_train += (trg == predicted).sum().item()
            training_accu.append((correct_train) / target_count)

            progress_bar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}, train accu: {sum(training_accu) / len(training_accu):.4f}", 
                refresh=True)
            progress_bar.update()
        
        return sum(training_loss) / len(training_loss), sum(training_accu) / len(training_accu)
    
    def evaluate(self, epoch, progress_bar):
        target_count, correct_train, train_acc = 0, 0, 0
        with torch.no_grad():
            # Set the model to eval
            self.model.eval()
            validation_loss = []
            # Loop through the validation batch
            for i, batch in enumerate(self.valid_iterator):
                src = batch.src
                trg = batch.trg
                # Forward pass
                output = self.model(src, trg, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                # Calculate Loss
                loss = self.criterion(output, trg)
                validation_loss.append(loss.item())
                
                _, predicted = torch.max(output.data, 1)
                target_count += trg.size(0)
                correct_train += (trg == predicted).sum().item()
                train_acc += (correct_train) / target_count

            progress_bar.set_postfix(
                epoch=f" {epoch}, val loss: {sum(validation_loss) / len(validation_loss):.4f}, val accu: {train_acc / len(self.valid_iterator):.4f}",
                refresh=False)
            progress_bar.close()
        
        return sum(validation_loss) / len(validation_loss), train_acc / len(self.valid_iterator)

    def train_model(self):

        for epoch in range(1, EPOCHS + 1):
            progress_bar = tqdm(total=len(self.train_iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

            # Train and evalute the model
            train_loss, train_accu = self.train(epoch, progress_bar)
            val_loss, val_accu = self.evaluate(epoch, progress_bar)

            self.show_train_metrics(epoch, train_loss, train_accu, val_loss, val_accu)
            self.save_train_metrics(epoch, train_loss, train_accu, val_loss, val_accu)
            self.save_model()

    def translate(self, sentence):

        # Convert each source token to integer values using the vocabulary
        tokens = ['<sos>'] + [token.lower() for token in sentence] + ['<eos>']
        src_indexes = [self.source.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

        self.model.eval()

        # Run the forward pass of the encoder
        hidden, cell = self.model.encoder(src_tensor)

        # Take the integer value of <sos> from the target vocabulary.
        trg_index = [self.target.vocab.stoi['<sos>']]
        next_token = torch.LongTensor(trg_index).to(device)

        outputs = []
        trg_indexes = []

        with torch.no_grad():
            # Use the hidden and cell vector of the Encoder and in loop
            # run the forward pass of the OneStepDecoder until some specified
            # step (say 50) or when <eos> has been generated by the model.
            for _ in range(30):
                output, hidden, cell = self.model.decoder.one_step_decoder(next_token, hidden, cell)

                # Take the most probable word
                next_token = output.argmax(1)

                trg_indexes.append(next_token.item())

                predicted = self.target.vocab.itos[output.argmax(1).item()]
                if predicted == '<eos>':
                    break
                else:
                    outputs.append(predicted)

        predicted_words = [self.target.vocab.itos[i] for i in trg_indexes]

        return predicted_words
    
    def translate_sentence(self, sentence: str) -> str:
        predicted_words = self.translate(sentence)
        return self.untokenized_translation(predicted_words)

    def untokenized_translation(self, translated_sentence_list) -> str:
        """
            Method to untokenuze the pedicted translation.
            Returning it on as an str.
        """
        translated_sentence_str = []
        for word in translated_sentence_list:
            if(word != "<eos>" and word != "<unk>"):
                translated_sentence_str.append(word)
        translated_sentence = TreebankWordDetokenizer().detokenize(translated_sentence_str)
        return self.grammar.check_sentence(translated_sentence)

    def console_model_test(self) -> None:
        os.system("clear")
        print("\n                     CV Creole Translator ")
        print("-------------------------------------------------------------\n")
        while True:
            Sentence = str(input(f'  Sentence (cv): '))
            translation = self.translate_sentence(Sentence)

            print(
                f'  Predicted (en): {translation}\n')