# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import number_translation_dataset

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
