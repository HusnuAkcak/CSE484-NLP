import math
import os
import secrets
import sys
from nltk import ngrams
import time
import pickle
from itertools import islice


# extract n grams
# prepares dictionary
# save it as binary files
def extract_and_store_n_grams(filename, n):
    file = open(filename)
    content = file.read()

    print(time.ctime())
    start = time.time()
    print(n, "-grams are being calculated...")
    _ngrams = ngrams(content.split(), n)
    end = time.time()
    print(n, "-grams are ready...")
    print(time.time())
    print(end - start)

    ngram_dict = {}

    print(time.ctime())
    start = time.time()
    print(n, "-dictionary is prepared...")
    for grams in _ngrams:
        ngram_dict.update({grams: int(ngram_dict.get(grams) or 0) + 1})
    print(n, "-dictionary is ready...")
    print(time.time())
    print(end - start)

    # print(ngram_dict)
    if sys.getsizeof(ngram_dict) > 500000:
        write_big_bin_files(ngram_dict, math.ceil(sys.getsizeof(ngram_dict) / 500000), str(n) + "gramsDict.pkl")
    else:
        dict_file = open(str(n) + "gramsDict.pkl", "wb")
        pickle.dump(ngram_dict, dict_file)

        dict_file.close()
        file.close()


# divide data into SIZE piece then returns them as array
def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def prepare_and_store_n_grams():
    for i in range(1, 6):
        extract_and_store_n_grams('wiki00_syllabicated', i)


# load ngram from given binary files, then calculate GT smoothing and store it to another file
def gt_smooth(filenames):
    content = {}
    for filename in filenames:
        file = open(filename, "rb")
        curr = pickle.load(file)
        content = dict(list(content.items()) + list(curr.items()))

    size = int(len(content) * (95 / 100))  # 95% of n grams

    c_dict = {}
    N = 0  # total number of grams
    # c values are calculated
    for i, (key, val) in enumerate(content.items()):
        if size < i:
            break

        N = N + val
        c_dict.update({val: (c_dict.get(val) or 0) + 1})

    # print(c_dict)
    # print(c_dict.get(1))

    GT_smoothed = {}
    # GT smoothing
    total_prob = 0
    for i, (key, val) in enumerate(content.items()):
        if size < i:
            break

        n_c = c_dict.get(val)
        n_c_plus1 = c_dict.get(val + 1) or 0
        c_star = (val + 1) * (n_c_plus1 / n_c)
        if c_star == 0:
            p_start_GT = content.get(key) / N
        else:
            p_start_GT = c_star / N

        total_prob = total_prob + p_start_GT
        GT_smoothed[key] = p_start_GT

    print('Test:', total_prob)
    GT_smoothed['UNSEEN'] = c_dict.get(1) / N
    # print('GT_smoothed table: ', GT_smoothed)
    print('N: ', N)
    print('N_1', c_dict.get(1))
    print('unseen prob: ', GT_smoothed.get('UNSEEN'))
    if sys.getsizeof(GT_smoothed) > 500000000:
        write_big_bin_files(GT_smoothed, math.ceil(sys.getsizeof(GT_smoothed) / 500000000),
                            filenames[0] + '_GT_smoothed.pkl')
    else:
        file = open(filenames[0] + '_GT_smoothed.pkl', 'wb')
        pickle.dump(GT_smoothed, file)
        file.close()


# ASSUMPTION: Sentences are consist of 25 syllables
def unigram_perplexity(ngram_filename, N, testData):
    sentence_len = 25
    # read ngram file to a dictionary
    file = open(ngram_filename, 'rb')
    ngram_dict = pickle.load(file)

    # calc N
    print('N = ', N)

    syllables = testData.split()

    # loop through each sentence
    i = 0
    while i < len(syllables) - sentence_len:
        curr_p = 0
        for j in range(sentence_len):
            # calc perplexity of sentence
            curr_p = curr_p + math.log(ngram_dict.get((syllables[i + j],)) or 1, math.e)
            # print(syllables[i+j], ' - ', ngram_dict.get((syllables[i+j],)))

        curr_p = (-1 / N) * curr_p
        for x in range(i, i + sentence_len):
            print(syllables[x], end=' ')
        print('')
        print('e^curr_p = ', math.exp(curr_p))
        i = i + sentence_len


# ASSUMPTION: Sentences are consist of 25 syllables
def bigram_perplexity(ngram_filename, N, testData):
    sentence_len = 25
    # read ngram file to a dictionary
    file = open(ngram_filename, 'rb')
    ngram_dict = pickle.load(file)

    # calc N
    print('N = ', N)

    syllables = testData.split()

    # loop through each sentence
    i = 0
    while i < len(syllables) - sentence_len -1:
        curr_p = 0
        for j in range(sentence_len):
            # calc perplexity of sentence
            curr_p = curr_p + math.log(ngram_dict.get((syllables[i + j], syllables[i+j+1], )) or 1, math.e)
            print(syllables[i+j], ',', syllables[i+j+1], ' - ', ngram_dict.get((syllables[i+j], syllables[i+j+1], )))

        curr_p = (-1 / N) * curr_p
        for x in range(i, i + sentence_len):
            print(syllables[x], end=' ')
        print('')
        print('e^curr_p = ', math.exp(curr_p))
        i = i + sentence_len


# ASSUMPTION: Sentences are consist of 25 syllables
def threegram_perplexity(ngram_filename, N, testData):
    sentence_len = 25
    # read ngram file to a dictionary
    file = open(ngram_filename, 'rb')
    ngram_dict = pickle.load(file)

    # calc N
    print('N = ', N)

    syllables = testData.split()

    # loop through each sentence
    i = 0
    while i < len(syllables) - sentence_len-2:
        curr_p = 0
        for j in range(sentence_len):
            # calc perplexity of sentence
            curr_p = curr_p + math.log(ngram_dict.get((syllables[i + j], syllables[i+j+1], syllables[i+j+2], )) or 1, math.e)
            print(syllables[i+j], ',', syllables[i+j+1], ',', syllables[i+j+2], ' - ',
                  ngram_dict.get((syllables[i+j], syllables[i+j+1], syllables[i+j+2], )))

        curr_p = (-1 / N) * curr_p
        for x in range(i, i + sentence_len):
            print(syllables[x], end=' ')
        print('')
        print('e^curr_p = ', math.exp(curr_p))
        i = i + sentence_len


def write_big_bin_files(_dict, sub_pieces, filename):
    print('sub_pieces', sub_pieces)
    step = len(_dict) / sub_pieces
    i = 0
    for item in chunks(_dict, int(step)):
        # print(item)
        print('i:', i)
        dict_file = open(filename + '_' + str(i + 1), "wb")
        pickle.dump(item, dict_file)
        i = i + 1
        dict_file.close()


def read_last_n_percent(filename, n):
    file = open(filename)

    size = os.path.getsize(filename)
    firstN = int(((100 - n) / 100) * size)

    file.seek(firstN)
    content = file.read()

    return content


def read_first_n_percent(filename, n):
    file = open(filename)

    size = os.path.getsize(filename)
    firstN = int((n / 100) * size)

    content = file.read(firstN)

    return content


def generate_random_sentences_from_unigram():

    file = open('1gramsDict.pkl_GT_smoothed.pkl', 'rb')
    unigram_dict = pickle.load(file)
    file.close()

    file = open('wiki00_syllabicated')
    wikidata = file.read().split()
    file.close()

    sentenceToGenerate = 10
    sentenceLen = 25
    for i in range(sentenceToGenerate):
        # pick a random syllable
        currSentence = []
        for j in range(sentenceLen):
            found = False
            while not found:
                rand_int = secrets.randbelow( len(wikidata)-1)
                syll = wikidata[rand_int]

                unigram_prob = unigram_dict.get((syll,)) or 0
                if unigram_prob > 1.0000000004533484e-04:
                    currSentence.append(syll)
                    found = True  # accept first syllable

        print(currSentence)


def generate_random_sentences_from_bigram():

    file = open('1gramsDict.pkl_GT_smoothed.pkl', 'rb')
    unigram_dict = pickle.load(file)
    file.close()

    file = open('2gramsDict.pkl_GT_smoothed.pkl', 'rb')
    bigram_dict = pickle.load(file)
    file.close()

    file = open('wiki00_syllabicated')
    wikidata = file.read().split()
    file.close()

    sentenceToGenerate = 10
    sentenceLen = 25
    for i in range(sentenceToGenerate):
        # pick a random syllable
        currSentence = []
        for j in range(sentenceLen):
            found = False
            while not found:
                rand_int = secrets.randbelow( len(wikidata)-1)
                syll = wikidata[rand_int]
                if j == 0:
                    unigram_prob = unigram_dict.get((syll,)) or 0
                    if unigram_prob > 1.0000000004533484e-04:
                        currSentence.append(syll)
                        found = True  # accept first syllable
                else:
                    bgramProb = bigram_dict.get((currSentence[j-1], syll, )) or 0
                    # print(bgramProb)
                    if bgramProb > 0.0100000004533484e-04:
                        found = True
                        currSentence.append(syll)

        print(currSentence)


def generate_random_sentences_from_threegram():

    file = open('3gramsDict.pkl_GT_smoothed.pkl', 'rb')
    threegram_dict = pickle.load(file)
    file.close()

    file = open('wiki00_syllabicated')
    wikidata = file.read().split()
    file.close()

    sentenceToGenerate = 10
    sentenceLen = 15
    for i in range(sentenceToGenerate):
        # pick a random syllable
        currSentence = []
        j = 0
        while j < sentenceLen:
            found = False
            while not found:
                if j == 0:
                    rand_1 = secrets.randbelow( len(wikidata) - 1)
                    rand_2 = secrets.randbelow(len(wikidata) - 1)
                    rand_3 = secrets.randbelow(len(wikidata) - 1)

                    syll1 = wikidata[rand_1]
                    syll2 = wikidata[rand_2]
                    syll3 = wikidata[rand_3]

                    threegramProb = threegram_dict.get((syll1, syll2, syll3,)) or 0
                    if threegramProb > 1.123456789e-04:
                        currSentence.append(syll1)
                        currSentence.append(syll2)
                        currSentence.append(syll3)
                        found = True
                        j = 3
                else:
                    rand_1 = secrets.randbelow( len(wikidata) - 1)
                    syll1 = wikidata[rand_1]

                    threegramProb = threegram_dict.get((currSentence[j-2], currSentence[j-1], syll1,)) or 0
                    if threegramProb > 1.123456789e-07:
                        currSentence.append(syll1)
                        found = True
                        j = j+1

        print(currSentence)


def generate_random_sentences_from_fourgram():

    file1 = open('4gramsDict_GT_smoothed_1.pkl', 'rb')
    file2 = open('4gramsDict_GT_smoothed_2.pkl', 'rb')
    file3 = open('4gramsDict_GT_smoothed_3.pkl', 'rb')
    file4 = open('4gramsDict_GT_smoothed_4.pkl', 'rb')
    fourgram_dict1 = pickle.load(file1)
    fourgram_dict2 = pickle.load(file2)
    fourgram_dict3 = pickle.load(file3)
    fourgram_dict4 = pickle.load(file4)

    fourgram_dict = dict(list(fourgram_dict1.items()) +
                          list(fourgram_dict2.items()) +
                          list(fourgram_dict3.items()) +
                          list(fourgram_dict4.items()))

    UNK_prob = 0  # fourgram_dict.get('UNSEEN')
    print('UNK:', UNK_prob)
    file1.close()
    file2.close()
    file3.close()
    file4.close()

    file = open('wiki00_syllabicated')
    wikidata = file.read().split()
    file.close()

    sentenceToGenerate = 101
    sentenceLen = 15
    for i in range(sentenceToGenerate):
        # pick a random syllable
        currSentence = []
        j = 0
        while j < sentenceLen:
            found = False
            while not found:
                if j == 0:
                    rand_1 = secrets.randbelow( len(wikidata) - 1)
                    rand_2 = secrets.randbelow(len(wikidata) - 1)
                    rand_3 = secrets.randbelow(len(wikidata) - 1)
                    rand_4 = secrets.randbelow(len(wikidata) - 1)

                    syll1 = wikidata[rand_1]
                    syll2 = wikidata[rand_2]
                    syll3 = wikidata[rand_3]
                    syll4 = wikidata[rand_4]

                    fourgramProb = fourgram_dict.get((syll1, syll2, syll3, syll4, )) or UNK_prob
                    if fourgramProb > 1.123456789e-07:
                        currSentence.append(syll1)
                        currSentence.append(syll2)
                        currSentence.append(syll3)
                        currSentence.append(syll4)
                        found = True
                        j = 4
                else:
                    rand_1 = secrets.randbelow( len(wikidata) - 1)
                    syll1 = wikidata[rand_1]

                    fourgramProb = fourgram_dict.get((currSentence[j-3], currSentence[j-2], currSentence[j-1], syll1,)) or UNK_prob
                    if fourgramProb > 1.123456789e-09:
                        currSentence.append(syll1)
                        found = True
                        j = j+1

        print(currSentence)


def generate_random_sentences_from_fivegram():

    #file1 = open('5gramsDict_GT_smoothed_1.pkl', 'rb')
    file2 = open('5gramsDict_GT_smoothed_2.pkl', 'rb')
    file3 = open('5gramsDict_GT_smoothed_3.pkl', 'rb')
    file4 = open('5gramsDict_GT_smoothed_4.pkl', 'rb')
    file5 = open('5gramsDict_GT_smoothed_5.pkl', 'rb')


    #fivegram_dict1 = pickle.load(file1)
    fivegram_dict2 = pickle.load(file2)
    fivegram_dict3 = pickle.load(file3)
    fivegram_dict4 = pickle.load(file4)
    fivegram_dict5 = pickle.load(file5)

    fivegram_dict = dict(#list(fivegram_dict1.items()) +
                         list(fivegram_dict2.items()) +
                          list(fivegram_dict3.items()) +
                          list(fivegram_dict4.items()) +
                          list(fivegram_dict5.items()))

    UNK_prob = 0  # fivegram_dict.get('UNSEEN')
    print('UNK:', UNK_prob)
    #file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()


    file = open('wiki00_syllabicated')
    wikidata = file.read().split()
    file.close()

    sentenceToGenerate = 101
    sentenceLen = 50
    for i in range(sentenceToGenerate):
        # pick a random syllable
        currSentence = []
        j = 0
        while j < sentenceLen:
            found = False
            while not found:
                if j == 0:
                    rand_1 = secrets.randbelow(len(wikidata) - 1)
                    rand_2 = secrets.randbelow(len(wikidata) - 1)
                    rand_3 = secrets.randbelow(len(wikidata) - 1)
                    rand_4 = secrets.randbelow(len(wikidata) - 1)
                    rand_5 = secrets.randbelow(len(wikidata) - 1)


                    syll1 = wikidata[rand_1]
                    syll2 = wikidata[rand_2]
                    syll3 = wikidata[rand_3]
                    syll4 = wikidata[rand_4]
                    syll5 = wikidata[rand_5]

                    fivegramProb = fivegram_dict.get((syll1, syll2, syll3, syll4, syll5 )) or UNK_prob
                    if fivegramProb > 1.123456789e-09:
                        currSentence.append(syll1)
                        currSentence.append(syll2)
                        currSentence.append(syll3)
                        currSentence.append(syll4)
                        currSentence.append(syll5)
                        found = True
                        j = 5
                else:
                    rand_1 = secrets.randbelow(len(wikidata) - 1)
                    syll1 = wikidata[rand_1]

                    fivegramProb = fivegram_dict.get((currSentence[j-4], currSentence[j-3], currSentence[j-2], currSentence[j-1], syll1,)) or UNK_prob
                    if fivegramProb > 0.123456789e-10:
                        currSentence.append(syll1)
                        found = True
                        j = j+1

        print(currSentence)


# gt_smooth(['1gramsDict.pkl'])
# gt_smooth(['2gramsDict.pkl'])
# gt_smooth(['3gramsDict.pkl'])
# gt_smooth(['4gramsDict.pkl'])
# gt_smooth(['5gramsDict_part1.pkl', '5gramsDict_part2.pkl', '5gramsDict_part3.pkl', '5gramsDict_part4.pkl', '5gramsDict_part5.pkl'])

# _testData = read_last_n_percent('wiki00_syllabicated', 4)
# _testData2 = read_last_n_percent('test', 100)

# unigram_perplexity('1gramsDict.pkl_GT_smoothed.pkl', N, _testData)
# bigram_perplexity('2gramsDict.pkl_GT_smoothed.pkl', N, _testData)
# threegram_perplexity('3gramsDict.pkl_GT_smoothed.pkl', N, _testData)


# generate_random_sentences_from_unigram()
# generate_random_sentences_from_bigram()
# generate_random_sentences_from_threegram()
# generate_random_sentences_from_fourgram()
generate_random_sentences_from_fivegram()

# file = open('2gramsDict.pkl', "rb")
# curr = pickle.load(file)
# print(curr)
