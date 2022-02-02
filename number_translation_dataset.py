
import math
import numpy as np
import torch
from torch.utils.data import Dataset

START_TOKEN = '^'
END_TOKEN = '$'

WORD_TOKENS = [START_TOKEN, END_TOKEN, 'and', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'trillion']

WORD_TO_INDEX = {WORD_TOKENS[i]: i for i in range(len(WORD_TOKENS))}
INDEX_TO_WORD = {i: WORD_TOKENS[i] for i in range(len(WORD_TOKENS))}

CHAR_TOKENS = [START_TOKEN, END_TOKEN, '零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
               '百', '千', '万', '亿']

CHAR_TO_INDEX = {CHAR_TOKENS[i]: i for i in range(len(CHAR_TOKENS))}
INDEX_TO_CHAR = {i: CHAR_TOKENS[i] for i in range(len(CHAR_TOKENS))}


class NumberTranslationDataset(Dataset):
    def __init__(self, size):
        """
        :param size: Size of the dataset.
        """
        super().__init__()
        self.nums = np.random.uniform(low=0, high=15, size=size)
        self.nums = np.round(np.exp(self.nums))
        self.nums = self.nums.astype(np.longlong)

        self.number_to_sent_converter = NumberToSentenceConverter()

    def __getitem__(self, index: int):
        n = self.nums[index]
        n_english = self.number_to_sent_converter.number_to_english(n)
        eng_tokens = self.number_to_sent_converter.tokenize_english_number(n_english)
        eng_tokens = torch.tensor(eng_tokens, dtype=torch.int64)
        eng_len = eng_tokens.shape[-1]
        eng_tokens = torch.hstack([eng_tokens, torch.ones(size=(20 - eng_len,), dtype=torch.long)])

        n_chinese = self.number_to_sent_converter.number_to_chinese(n)
        ch_tokens = self.number_to_sent_converter.tokenize_chinese_number(n_chinese)
        ch_tokens = torch.tensor(ch_tokens, dtype=torch.int64)

        return n, eng_tokens, ch_tokens

    def __len__(self):
        return len(self.nums)


class NumberToSentenceConverter:
    def __init__(self):
        # Dictionary converting key numbers to corresponding english word
        self.num_to_word = {}
        self.init_num_to_words_dict()

        self.num_to_char = {}
        self.init_num_to_chars_dict()

    def tokenize_english_number(self, eng_number):
        eng_number = START_TOKEN + ' ' + eng_number + ' ' + END_TOKEN
        tokens = [WORD_TO_INDEX[word] for word in eng_number.split()]
        return tokens

    def tokenize_chinese_number(self, ch_number):
        ch_number = START_TOKEN + ch_number + END_TOKEN
        tokens = [CHAR_TO_INDEX[c] for c in list(ch_number)]
        return tokens

    def init_num_to_chars_dict(self):
        ones = list('零一二三四五六七八九')
        
        for i in range(len(ones)):
            self.num_to_char[i] = ones[i]

        self.num_to_char[10] = '十'
        self.num_to_char[100] = '百'
        self.num_to_char[1000] = '千'
        self.num_to_char[10 ** 4] = '万'
        self.num_to_char[10 ** 8] = '亿'

    def number_to_chinese(self, n, is_suffix=False):
        if n == 0:
            if is_suffix:
                sentence = ''
            else:
                sentence = self.num_to_char[n]
        elif n < 10:
            sentence = self.num_to_char[n]
        elif n < 10000:
            num_digits = math.floor(math.log10(n)) + 1
            # largest power of 10 at most n
            pow_of_10 = 10 ** (num_digits - 1)
            leading_digit = n // pow_of_10
            remainder = n - leading_digit * pow_of_10
            sentence = self.num_to_char[leading_digit] + self.num_to_char[pow_of_10]
            if remainder > 0:
                if remainder < pow_of_10 / 10:
                    sentence += self.num_to_char[0]
            sentence += self.number_to_chinese(remainder, is_suffix=True)
        else:
            num_digits = math.floor(math.log10(n)) + 1
            # largest power of 10000 less than n
            power_of_10000 = (num_digits - 1) // 4

            # split off 145789 into leading=14 and trailing=5789 (groups of 4)
            m = 10000 ** power_of_10000
            leading = n // m
            trailing = n % m
            sentence = self.number_to_chinese(leading) + self.num_to_char[m]
            if trailing > 0:
                # 100005 would be 十万零五 need to add 零 if there's a 0 in between
                if trailing < m / 10:
                    sentence += self.num_to_char[0]
                sentence += self.number_to_chinese(trailing, is_suffix=True)

        return sentence

    def init_num_to_words_dict(self):
        teens = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen ' \
                'fifteen sixteen seventeen eighteen nineteen'
        teens = teens.split()
        for i in range(20):
            self.num_to_word[i] = teens[i]

        tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()
        for i in range(len(tens)):
            n = (i + 2) * 10
            self.num_to_word[n] = tens[i]

        self.num_to_word[100] = 'hundred'
        self.num_to_word[1000] = 'thousand'
        self.num_to_word[1000 ** 2] = 'million'
        self.num_to_word[1000 ** 3] = 'billion'
        self.num_to_word[1000 ** 4] = 'trillion'

    def number_to_english(self, n, is_suffix=False):
        """Returns string of english words reading out number `n`.

           :param is_suffix: if True, then 0 -> '', else 0 -> 'zero'. Used internally for recursive calls.
        """
        if n == 0:
            if is_suffix:
                sentence = ''
            else:
                sentence = self.num_to_word[n]
        elif n < 20:
            sentence = 'and ' if is_suffix else ''
            sentence += self.num_to_word[n]
        elif n < 100:
            leading_digit = n // 10
            remainder = n - leading_digit * 10
            sentence = 'and ' if is_suffix else ''
            sentence += self.num_to_word[leading_digit * 10]
            if remainder > 0:
                sentence += ' ' + self.number_to_english(remainder, is_suffix=False)
        elif n < 1000:
            leading_digit = n // 100
            remainder = n - leading_digit * 100
            sentence = self.num_to_word[leading_digit] + ' ' + self.num_to_word[100] + ' ' + \
                self.number_to_english(remainder, is_suffix=True)
        else:
            num_digits = math.floor(math.log10(n)) + 1
            # largest power of 1000 less than n
            power_of_1000 = (num_digits - 1) // 3

            # split off 12,345,789 into leading=12 and trailing=345,789
            m = 1000 ** power_of_1000
            leading = n // m
            trailing = n % m
            sentence = self.number_to_english(leading, is_suffix=False) + ' ' + self.num_to_word[m]
            if trailing > 0:
                sentence += ' ' + self.number_to_english(trailing, is_suffix=True)

        return sentence.strip()
