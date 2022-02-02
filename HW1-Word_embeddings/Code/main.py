import pickle
import struct
import numpy as np
from turkishnlp import detector
import re


def syllabicate(input_file_name, output_file_name):
    file = open(input_file_name, encoding="utf8")
    all_books = file.read()

    obj = get_object('nlp_obj')

    print("books are syllabicated...")
    syllable_2 = obj.syllabicate(all_books)
    print(syllable_2)

    with open(output_file_name, 'w', encoding='utf8') as f:
        for syllable in syllable_2:
            try:
                f.write(" ".join(re.findall("[a-zA-ZğĞçÇşŞüÜöÖıİ]+", syllable)) + ' ')
            except BaseException as error:
                print(error.__str__() + "\n")


def save_nlp_object(filename):
    obj = detector.TurkishNLP()
    obj.download()
    obj.create_word_set()

    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_object(filename):
    file = open(filename, 'rb')
    return pickle.load(file)


def filter_alphabetic_chars(filename):
    file = open(filename, 'r', encoding="utf8")
    all_text = file.read()

    output_file = open('FILTERED' + filename, 'w', encoding="utf8")
    for word in all_text.split(' '):
        output_file.write(" ".join(re.findall("[a-zA-ZğĞçÇşŞüÜöÖıİ]+", word)) + ' ')


def recalc_vec_with_word_and_syllable(word_vec_file_name, syllable_vec_file_name):
    word_vec_arr = []
    syllable_vec_arr = []

    with open(word_vec_file_name, 'r', encoding='UTF-8') as file:
        for line in file:
            word_vec_arr.append(line.split(' '))

    with open(syllable_vec_file_name, 'r', encoding='UTF-8') as file:
        for line in file:
            syllable_vec_arr.append(line.split(' '))

    final_file_txt = open('from_syllable_to_word_txt', 'w', encoding='utf-8')
    final_file_bin = open('from_syllable_to_word_bin', 'wb')
    num_of_words = word_vec_arr[0][0]
    len_of_vector = word_vec_arr[0][1]

    final_file_bin.write(bytes(num_of_words + " " + len_of_vector, encoding='utf-8'))
    final_file_txt.write(num_of_words + " " + len_of_vector)

    nlp_obj = get_object('nlp_obj')
    all_syllables = [row[0] for row in syllable_vec_arr]
    curr_syllable_vectors = []
    for i in range(1, len(word_vec_arr)):
        curr = word_vec_arr[i]
        curr_syllables = nlp_obj.syllabicate(curr[0])
        print(curr[0])
        for syllable in curr_syllables:
            s_ind = get_index(all_syllables, syllable)
            curr_syllable_vectors.append(np.array(syllable_vec_arr[s_ind][1:-1]).astype(np.float64))
            # print(syllable, ' index:', s_ind)

        new_vector = np.add.reduce(curr_syllable_vectors)
        final_file_bin.write(bytes(curr[0]+" ", encoding='utf-8'))
        for val in new_vector:
            final_file_bin.write(struct.pack('f', val))
        final_file_bin.write(bytes("\n", encoding='utf-8'))

        final_file_txt.write(curr[0]+" ")
        final_file_txt.write(" ".join(str(item) for item in new_vector)+"\n")
        curr_syllable_vectors = []
    final_file_bin.close()
    final_file_txt.close()


def get_index(arr, target):
    try:
        return arr.index(target)
    except:
        return -1


# syllabicate("_all_books.txt", "_all_books_syllabicated.txt")
# filter_alphabetic_chars('_all_books.txt')
recalc_vec_with_word_and_syllable('vectors_all_books', 'vectors_all_books_syllabicated')


