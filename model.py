
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union
import number_translation_dataset as num_dataset
from torch.utils.data import DataLoader

class NumberTranslationModel(nn.Module):
    def __init__(self, num_input_tokens: int, num_output_tokens: int, input_embedding_dim: int,
                 output_embedding_dim: int, encoder_lstm_hidden_dim: int, encoder_num_layers: int,
                 bidirectional: bool, input_start_token_index: int,
                 output_start_token_index: int, input_end_token_index: int, output_end_token_index: int):
        super().__init__()
        self._num_input_tokens = num_input_tokens
        self._num_output_tokens = num_output_tokens
        self._input_embedding_dim = input_embedding_dim
        self._encoder_lstm_hidden_dim = encoder_lstm_hidden_dim
        self._encoder_num_layers = encoder_num_layers
        self._bidirectional = bidirectional

        self._input_start_token_index = input_start_token_index
        self._output_start_token_index = output_start_token_index
        self._input_end_token_index = input_end_token_index
        self._output_end_token_index = output_end_token_index

        self._input_token_embedding = nn.Embedding(num_input_tokens, input_embedding_dim)
        #self._encoder_lstm = nn.LSTMCell(input_embedding_dim, encoder_lstm_hidden_dim)
        self._encoder_lstm = nn.LSTM(input_size=input_embedding_dim,
                                     hidden_size=encoder_lstm_hidden_dim,
                                     num_layers=encoder_num_layers,
                                     bidirectional=bidirectional)

        self._output_token_embedding = nn.Embedding(num_output_tokens, output_embedding_dim)
        D = 2 if bidirectional else 1
        decoder_lstm_hidden_dim = D * encoder_num_layers * encoder_lstm_hidden_dim
        self._decoder_lstm = nn.LSTMCell(output_embedding_dim, decoder_lstm_hidden_dim)

        # right now the decoder is only a single layer.
        #self._decoder_lstm = nn.LSTM(input_size=output_embedding_dim,
        #                             hidden_size=2*encoder_num_layers*encoder_lstm_hidden_dim)

        # linearly project output of decoder lstm to logits
        self._decoder_proj = nn.Linear(decoder_lstm_hidden_dim, num_output_tokens)

    def get_init_state(self, batch_size, device):
        """Returns initial state of encoder LSTM."""
        D = 2 if self._bidirectional else 1
        num_layers = self._encoder_num_layers
        hidden = torch.zeros(size=(D * num_layers, batch_size, self._encoder_lstm_hidden_dim), device=device)
        cell = torch.zeros(size=(D * num_layers, batch_size, self._encoder_lstm_hidden_dim), device=device)
        return hidden, cell

    def greedy_predict(self, input_tokens: List[torch.Tensor], device, max_output_len, index_to_words):
        # lstm state of encoder is used as input decoder state
        decoder_state = self.forward_encoder(input_tokens, device)
        batch_size = len(input_tokens)

        token_idx = 0
        stop_cond_reached = False
        # list of output tensors, each of size batch_size. Initialize with start token index.
        output_tokens = [torch.full(size=(batch_size,), fill_value=self._output_start_token_index,
                                    dtype=torch.long, device=device)]
        log_probs = [torch.zeros(size=(batch_size,), dtype=torch.float32, device=device)]

        while not stop_cond_reached and token_idx < max_output_len:
            output_token_embedding = self._output_token_embedding(output_tokens[-1])
            decoder_state = self._decoder_lstm(output_token_embedding, decoder_state)
            hidden, cell = decoder_state
            next_token_logits = self._decoder_proj(cell)
            # Greedy choice for each token.
            greedy_tokens = torch.argmax(next_token_logits, dim=1)
            output_tokens.append(greedy_tokens)
            dist = torch.distributions.categorical.Categorical(logits=next_token_logits)
            log_probs.append(dist.log_prob(greedy_tokens))
            token_idx += 1

        def convert_index_to_words(idx: int):
            return index_to_words[idx]

        token_tensor = torch.stack(output_tokens, dim=1)
        sentences = []
        for i in range(batch_size):
            sentences.append(''.join([index_to_words[idx.item()] for idx in token_tensor[i, :]]))
        return {'token_tensor': token_tensor, 'log_probs': log_probs}

    def forward_encoder(self, input_tokens: List[torch.Tensor], device):
        """
        Computes forward pass over the encoder.
        :param input_tokens: each tensor in list represents one sentence
        :param device:
        :return: (hidden, cell) state output of encoder, each has shape
        """
        input_tensor = nn.utils.rnn.pad_sequence(input_tokens, batch_first=False,
                                                 padding_value=self._input_end_token_index).to(device=device)
        batch_size = input_tensor.shape[1]
        input_max_len = input_tensor.shape[0]

        state = self.get_init_state(batch_size, device)

        # token_embedding_tensor has shape (input_max_len, batch_size, embedding_dim)
        token_embedding_tensor = self._input_token_embedding(input_tensor)

        state = self._encoder_lstm(token_embedding_tensor)

        #for i in range(input_max_len):
        #    token_embedding = self._input_token_embedding(input_tensor[:, i])
        #    state = self._encoder_lstm(token_embedding, state)

        # each of hidden, cell has shape (D * num_layers, N, H_in)
        # convert to (1, N, D * num_layers * H_in) for input into the decoder LSTM
        output, h_and_c = state

        h_and_c = list(h_and_c)
        for i in range(2):
            h_and_c[i] = torch.swapaxes(h_and_c[i], 0, 1)
            h_and_c[i] = torch.reshape(h_and_c[i], (h_and_c[i].shape[0], -1))

        return h_and_c

    def forward(self, input_tokens: List[torch.Tensor], output_tokens, device):
        """ Forward pass in teacher forcing mode.
        :param input_tokens: batch of (possibly different size) token tensors
        :param output_tokens: batch of (possibly different size) token tensors
        :return:
        """
        # encoder lstm state is used as input state to the decoder
        state = self.forward_encoder(input_tokens, device)

        # we operate in teacher forcing mode where we use output_tokens as inputs to the
        # decoder, otherwise we use decoder output as decoder inputs of next time step.
        output_tensor = nn.utils.rnn.pad_sequence(output_tokens, batch_first=True,
                                                  padding_value=self._input_end_token_index).to(device=device)
        output_batch_size = output_tensor.shape[0]
        output_max_len = output_tensor.shape[1]
        decoder_state = state
        # list of (batch of) tokens output by decoder
        decoded_tokens = []
        # log probabilities of greedy choice for each output step
        log_probs = []
        for i in range(output_max_len - 1):
            output_token_embedding = self._output_token_embedding(output_tensor[:, i])
            decoder_state = self._decoder_lstm(output_token_embedding, decoder_state)
            hidden, cell = decoder_state
            next_token_logits = self._decoder_proj(cell)
            # Greedy choice for each token.
            greedy_tokens = torch.argmax(next_token_logits, dim=1).to(device=device)
            decoded_tokens.append(greedy_tokens)
            dist = torch.distributions.categorical.Categorical(logits=next_token_logits)
            log_probs.append(dist.log_prob(output_tensor[:, i + 1]))
        cum_log_probs = torch.sum(torch.stack(log_probs, dim=0), dim=0)

        return {'output_tokens': decoded_tokens, 'cum_log_probs': cum_log_probs}

    def _encoder_one_step(self, input_hidden_state, token: torch.Tensor):
        """Performs one step of the encoder applied to a batch of tokens.

        :param input_hidden_state: (hidden, cell), each of shape (batch_size x self._encoder_lstm_hidden_dim)
        :param token: shape (batch x 1)
        :return: a tuple of (hidden, cell) states after the step is performed
        """
        token_embedding = self._input_token_embedding(token)
        hidden, cell = self._encoder_lstm(token_embedding, input_hidden_state)
        return hidden, cell

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    # test the model

    # english to chinese
    model = NumberTranslationModel(num_input_tokens=len(num_dataset.WORD_TOKENS),
                                   num_output_tokens=len(num_dataset.CHAR_TOKENS),
                                   input_embedding_dim=10,
                                   output_embedding_dim=10,
                                   encoder_lstm_hidden_dim=30,
                                   input_end_token_index=num_dataset.WORD_TO_INDEX[num_dataset.END_TOKEN],
                                   output_end_token_index=num_dataset.CHAR_TO_INDEX[num_dataset.END_TOKEN])
    dataset = num_dataset.NumberTranslationDataset(size=10)
    device = 'cpu'
    n, eng_tokens, ch_tokens = dataset[0]
    print(f'eng_tokens: {eng_tokens}')
    print(f'ch_tokens: {ch_tokens}')
    # view single example as a batch
    #eng_tokens = eng_tokens.view(1, -1)
    #ch_tokens = ch_tokens.view(1, -1)

    def collate_fn(x):
        return x

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    for item in dataloader:
        print(item)
    model_output = model(input_tokens=[eng_tokens, eng_tokens], device=device, output_tokens=[ch_tokens, ch_tokens])
    print(model_output)