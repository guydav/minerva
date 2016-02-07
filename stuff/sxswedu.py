print 'Importing...'
import nltk
import csv
from tqdm import tqdm
from formal.utils import PriorityQueue
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


COMMON_100_WORDS = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on',
                        'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
                        'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
                        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make',
                        'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
                        'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come',
                        'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
                        'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
                        'is', 'are', 'has'])
PUNCTUATION = set(['.', ',', '(', ')', ':', '?', '!'])
INPUT_FILE = '/Users/guydavidson/PycharmProjects/Minerva/stuff/SXSWedu_and_Minerva_Student_PanelPicker_Data_Project.csv'
OUTPUT_FILE = 'cleaned_output.csv'
LIMIT = 100


def ascii_hack(string):
    return ''.join([c for c in string if (ord(c) < 128 and c not in PUNCTUATION)])


def clean_words(string):
    asciified = ascii_hack(string)
    clean = ' '.join([word for word in asciified.split() if word.lower() not in COMMON_100_WORDS])
    return clean


def read_data(path):
    descriptions = []
    learning_objectives = []
    tags = []

    with open(path) as data_file:
        reader = csv.reader(data_file.readlines())

        print 'Reading input:'

        for line in tqdm(reader):
            descriptions.append(clean_words(line[5]))
            learning_objectives.append('\n'.join([clean_words(line[obj_col]) for obj_col in (6, 7, 8)]))
            tags.append('\n'.join([clean_words(line[tag_col]) for tag_col in (9, 10, 11)]))

            # for obj_column in (6, 7, 8):
            #     learning_objectives.append(clean_words(line[obj_column]))
            # for tag_column in (9, 10, 11):
            #     tags.append(clean_words(line[tag_column]))

    return descriptions, learning_objectives, tags


def process_ngram(ngram_generator, limit=LIMIT):
    result = {}

    print 'Processing ngrams:'
    for ngram in tqdm(ngram_generator):
        if ngram not in result:
            result[ngram] = 1

        else:
            result[ngram] += 1

    print 'Removing results with n=1:'
    to_remove = set()
    for ngram in tqdm(result):
        if result[ngram] <= 1:
            to_remove.add(ngram)

    for ngram in to_remove:
        del result[ngram]

    sorted_result = sorted(result.items(), lambda x, y: result[x[0]].__cmp__(result[y[0]]), reverse=True)

    if limit:
        return sorted_result[:limit]

    else:
        return sorted_result


def process_data_set(data_set):
    words = nltk.word_tokenize('\n'.join(data_set))
    top_words = process_ngram(words)
    top_bigrams = process_ngram(nltk.bigrams(words))
    top_trigrams = process_ngram(nltk.trigrams(words))
    return top_words, top_bigrams, top_trigrams


def process_data(descriptions, learning_objectives, tags):
    output = []
    output.extend(process_data_set(descriptions))
    output.extend(process_data_set(learning_objectives))
    output.extend(process_data_set(tags))
    output.extend(process_data_set(descriptions + learning_objectives + tags))
    return output


HEADERS = ('Description Words', 'Description Bigrams', 'Description Trigrams', 'Objective Words', 'Objective Bigrams',
           'Objective Trigrams', 'Tag Words', 'Tag Bigrams', 'Tag Trigrams', 'Everything Words', 'Everything Bigrams',
           'Everything Trigrams')


def write_output(output, path):
    with open(path, 'w') as output_file:
        writer = csv.writer(output_file)

        for index in xrange(len(output)):
            current_output = output[index]
            current_output.insert(0, HEADERS[index])
            writer.writerow(current_output)


def words_and_similarities(data_set, n=10):
    words = nltk.word_tokenize('\n'.join(data_set).lower())
    top_words = process_ngram(words)
    ci = nltk.text.ContextIndex(words)

    for word_tuple in top_words[:n]:
        word = word_tuple[0]
        similar_words = ci.word_similarity_dict(word)
        pq = PriorityQueue()
        for similar_word in similar_words:
            pq.put(similar_word, similar_words[similar_word])

        top_matches = pq.get_many(n, False, True)
        print word, top_matches[1:]


def gensim_tokenizing(data_set):
    data_set = [text.lower() for text in data_set]
    tokenized_data = [nltk.word_tokenize(text) for text in data_set]
    top_word_tuples = process_ngram(nltk.word_tokenize('\n'.join(data_set)), limit=None)
    recurring_words = set([word_tuple[0] for word_tuple in top_word_tuples])
    tokenized = [[token for token in tokenized_item if token in recurring_words] for tokenized_item in tokenized_data]

    tokenized_dict = gensim.corpora.Dictionary(tokenized)
    corpus = [tokenized_dict.doc2bow(text_tokens) for text_tokens in tokenized_data]
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=tokenized_dict, num_topics=5)
    lsi.show_topics()
    print lsi[corpus[1]]
    # lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=tokenized_dict,
    #                                       num_topics=10, update_every=0, passes=10)
    # lda.print_topics(10)

def main():
    # descriptions, learning_objectives, tags = read_data(INPUT_FILE)
    # output = process_data(descriptions, learning_objectives, tags)
    # write_output(output, OUTPUT_FILE)

    descriptions, learning_objectives, tags = read_data(INPUT_FILE)
    gensim_tokenizing(descriptions)

if __name__ == '__main__':
    main()
