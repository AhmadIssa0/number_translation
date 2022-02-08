# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import number_translation_dataset
from model import NumberTranslationModel
from trainer import load_model_from_file
import torch

def english_to_chinese(model: NumberTranslationModel, english_sentence, device='cuda', padding_len=30):
    converter = number_translation_dataset.NumberToSentenceConverter()
    english_tokens = converter.tokenize_english_number(english_sentence)
    eng_tokens_tensor = torch.tensor(english_tokens, dtype=torch.int64, device=device)
    eng_len = eng_tokens_tensor.shape[0]
    eng_tokens_tensor = torch.hstack([eng_tokens_tensor,
                                      torch.ones(size=(padding_len - eng_len,), dtype=torch.long, device=device)])
    output = model.greedy_predict([eng_tokens_tensor], max_output_len=30, device=device,
                                  index_to_words=number_translation_dataset.INDEX_TO_CHAR)
    ch_token_tensor = output['token_tensor'][0]
    ch_chars = converter.chinese_tokens_to_chars(ch_token_tensor.tolist())
    # remove initial character '^' and everything after end char '$'
    ch_chars = ch_chars[1:]
    end_token_idx = ch_chars.find(number_translation_dataset.END_TOKEN)
    if end_token_idx != -1:
        ch_chars = ch_chars[:end_token_idx]
    return ch_chars

def main():
    model = load_model_from_file('model.pt', 'cuda')
    print(english_to_chinese(model, 'one hundred'))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    """
    print_hi('PyCharm')
    ntd = number_translation_dataset.NumberToSentenceConverter()
    print(ntd.number_to_english(3989))
    print(48038206)
    print(ntd.number_to_chinese(80800))
    print(ntd.num_to_char.values())
    ntd = number_translation_dataset.NumberTranslationDataset(10)
    print(ntd.nums)
    n, eng, ch = ntd.__getitem__(0)
    print(n)
    print([number_translation_dataset.INDEX_TO_WORD[i.item()] for i in eng])
    """

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
