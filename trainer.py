import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union
import number_translation_dataset
import number_translation_dataset as num_dataset
from torch.utils.data import DataLoader
from model import NumberTranslationModel


def fit(model, dataloader, epochs, optimizer: torch.optim.Optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on device', device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=20)
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        compute_acc = True if epoch % 20 == 0 else False

        training_metrics = eval(model, dataloader, compute_acc=compute_acc)
        print(f'training set log prob:', training_metrics)

        for data_batch in dataloader:
            optimizer.zero_grad()
            nums, eng_tokens, ch_tokens = tuple(zip(*data_batch))
            output = model(eng_tokens, ch_tokens, device=device)
            loss = -torch.mean(output['cum_log_probs'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step(training_metrics['avg_cum_log_prob'])

def accuracy(prediction_tensor, true_output_tokens):
    """
    Returns the number of rows for which the two inputs are the same.
    prediction_tensor is padded, ignores anything after the end_token.
    :param prediction_tensor:
    :param true_output_tokens:
    :return:
    """
    num_matching = 0
    for i in range(len(true_output_tokens)):
        # single example sentence, which is a tensor
        true_tokens = true_output_tokens[i]
        matches = True
        for j in range(true_tokens.shape[0]):
            if prediction_tensor[i, j].item() != true_tokens[j]:
                matches = False
                break
        if matches:
            num_matching += 1
    return num_matching

def eval(model, dataloader, compute_acc=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    with torch.no_grad():
        cum_log_prob = 0.0
        num_examples = 0
        correct_predictions = 0
        for data_batch in dataloader:
            nums, eng_tokens, ch_tokens = tuple(zip(*data_batch))
            output = model(eng_tokens, ch_tokens, device=device)
            cum_log_prob += torch.sum(output['cum_log_probs'])
            num_examples += len(data_batch)

            if compute_acc:
                prediction_tensor = model.greedy_predict(input_tokens=eng_tokens, device=device, max_output_len=25,
                                              index_to_words=number_translation_dataset.INDEX_TO_CHAR)['token_tensor']
                correct_predictions += accuracy(prediction_tensor, ch_tokens)
                """
                print('prediction tensor:')
                print(prediction_tensor)
                print('ch_tokens:')
                print(ch_tokens)
                print('accuracy:', accuracy(prediction_tensor, ch_tokens))
                """

        # return average log probability of output
        metrics = {}
        metrics['avg_cum_log_prob'] = cum_log_prob / num_examples

        if compute_acc:
            metrics['predictions_acc'] = correct_predictions / num_examples
        return metrics


if __name__ == '__main__':
    # test the model

    # english to chinese
    model = NumberTranslationModel(num_input_tokens=len(num_dataset.WORD_TOKENS),
                                   num_output_tokens=len(num_dataset.CHAR_TOKENS),
                                   input_embedding_dim=10,
                                   output_embedding_dim=10,
                                   encoder_lstm_hidden_dim=50,
                                   input_start_token_index=num_dataset.WORD_TO_INDEX[num_dataset.START_TOKEN],
                                   input_end_token_index=num_dataset.WORD_TO_INDEX[num_dataset.END_TOKEN],
                                   output_start_token_index=num_dataset.CHAR_TO_INDEX[num_dataset.START_TOKEN],
                                   output_end_token_index=num_dataset.CHAR_TO_INDEX[num_dataset.END_TOKEN])
    dataset = num_dataset.NumberTranslationDataset(size=10000)
    device = 'cuda'


    def collate_fn(x):
        return x


    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model = model.to(device='cuda')

    fit(model, dataloader, epochs=10000, optimizer=optimizer)
    """
    for i in range(1000):
        fit(model, dataloader, epochs=20, optimizer=optimizer)

        print(eval(model, dataloader, compute_acc=True))

        n, en, ch = dataset[0]
        
        predictions = model.greedy_predict(input_tokens=(en,), device='cuda', max_output_len=25,
                                           index_to_words=number_translation_dataset.INDEX_TO_CHAR)
        print('n=', n)
        print('english:', en)
        print('chinese:', ch)
        print(predictions)
        out = model(input_tokens=[en], output_tokens=[ch], device='cuda')
        print('cum log prob:', out['cum_log_probs'])
    """
