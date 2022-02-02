import pickle
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
                f.write(" ".join(syllable) + ' ')
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
        word = word.replace('ğ', 'g')
        word = word.replace('Ğ', 'G')
        word = word.replace('ç', 'c')
        word = word.replace('Ç', 'C')
        word = word.replace('ş', 's')
        word = word.replace('Ş', 'S')
        word = word.replace('ü', 'u')
        word = word.replace('Ü', 'U')
        word = word.replace('ö', 'o')
        word = word.replace('Ö', 'O')
        word = word.replace('ı', 'i')
        word = word.replace('I', 'i')
        word = word.replace('İ', 'i')

        word = re.findall("[a-zA-ZğĞçÇşŞüÜöÖıİ]+", word)

        output_file.write(" ".join(word) + ' ')


syllabicate("FILTEREDwiki_00", "wiki00_syllabicated")

#filter_alphabetic_chars('wiki_00')

