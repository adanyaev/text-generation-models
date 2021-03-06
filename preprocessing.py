import numpy as np
import re


class Preprocessing:

    @staticmethod
    def read_dataset(file):
        s11 = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя.,- '
        letters = [i for i in s11]

        with open(file, 'r', encoding='utf-8') as f:
            raw_text = f.readlines()

        raw_text = [line.lower() for line in raw_text]

        text_string = ''
        for line in raw_text:
            text_string += line.strip() + ' '

        text = list()
        for char in text_string:
            text.append(char)

        text = [char for char in text if char in letters]
        with open('text_list.txt', 'w', encoding='utf-8') as f:
            f.write(''.join(text))
        return text

    @staticmethod
    def create_dictionary(text):

        char_to_idx = dict()
        idx_to_char = dict()

        idx = 0
        for char in text:
            if char not in char_to_idx.keys():
                char_to_idx[char] = idx
                idx_to_char[idx] = char
                idx += 1

        print("Vocab: ", len(char_to_idx))

        return char_to_idx, idx_to_char

    @staticmethod
    def build_sequences_target(text, char_to_idx, window):

        x = list()
        y = list()
        i = 0
        raw_x = []
        raw_y = []
        while i + window < len(text):
            try:
                sequence = text[i:i + window]
                raw_x.append(sequence)
                sequence = [char_to_idx[char] for char in sequence]

                target = text[i+1:i + window + 1]
                raw_y.append(target)
                target = [char_to_idx[char] for char in target]
                i += window + 1
                x.append(sequence)
                y.append(target)
            except:
                pass

        with open("my_sequences.txt", 'w', encoding='utf-8') as f:
            for i in range(len(x)):
                f.write("{}:{}\n".format(raw_x[i], raw_y[i]))

        x = np.array(x)
        y = np.array(y)

        return x, y

