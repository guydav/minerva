import nltk
import csv
from tqdm import tqdm
from formal.utils import PriorityQueue
import gensim
from sklearn import cluster, metrics
from scipy.spatial import distance

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


COMMON_100_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on',
                    'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
                    'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
                    'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make',
                    'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
                    'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come',
                    'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
                    'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
                    'is', 'are', 'has'}
PUNCTUATION = {'.', ',', '(', ')', ':', '?', '!'}
INPUT_FILE = 'SXSWedu_and_Minerva_Student_PanelPicker_Data_Project.csv'
OUTPUT_FILE = 'cleaned_output.csv'
LIMIT = 100


def ascii_hack(string):
    return ''.join([c for c in string if (ord(c) < 128 and c not in PUNCTUATION)])


def clean_words(string):
    asciified = ascii_hack(string)
    clean = ' '.join([word for word in asciified.split() if word.lower() not in COMMON_100_WORDS])
    return clean


def read_data(path):
    titles = []
    descriptions = []
    learning_objectives = []
    tags = []

    with open(path) as data_file:
        reader = csv.reader(data_file.readlines())

        print 'Reading input:'

        for line in tqdm(reader):
            titles.append(clean_words(line[0]))
            descriptions.append(clean_words(line[5]))
            learning_objectives.append('\n'.join([clean_words(line[obj_col]) for obj_col in (6, 7, 8)]))
            tags.append('\n'.join([clean_words(line[tag_col]) for tag_col in (9, 10, 11)]))

            # for obj_column in (6, 7, 8):
            #     learning_objectives.append(clean_words(line[obj_column]))
            # for tag_column in (9, 10, 11):
            #     tags.append(clean_words(line[tag_column]))

    return titles, descriptions, learning_objectives, tags


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
    tokenized_data = [nltk.word_tokenize(text) for text in data_set]
    top_word_tuples = process_ngram(nltk.word_tokenize('\n'.join(data_set)), limit=0)
    recurring_words = set([word_tuple[0] for word_tuple in top_word_tuples])
    tokenized = [[token for token in tokenized_item if token in recurring_words] for tokenized_item in tokenized_data]

    tokenized_dict = gensim.corpora.Dictionary(tokenized)
    corpus = [tokenized_dict.doc2bow(text_tokens) for text_tokens in tokenized_data]
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=tokenized_dict, num_topics=5)
    lsi.show_topics()

    for desc in corpus[:10]:
        print lsi[desc]

    # lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=tokenized_dict,
    #                                       num_topics=10, update_every=0, passes=10)
    # lda.print_topics(10)

    corpus_topics = [[tup[1] for tup in lsi[desc]] for desc in corpus if desc]
    # for i in xrange(len(corpus_topics)):
    #     ct = corpus_topics[i]
    #     if len(ct) != 5:
    #         print ct, corpus[i]

    # for potential_k in tqdm(xrange(2,11)):
    #     kmeans = cluster.KMeans(n_clusters=potential_k)
    #     kmeans.fit(corpus_topics)
    #     fitted = kmeans.predict(corpus_topics)
    #     matching_cluster_centers = [kmeans.cluster_centers_[fitted[i]] for i in xrange(len(fitted))]
    #     sse = metrics.mean_squared_error(corpus_topics, matching_cluster_centers)
    #     distances = [distance.euclidean(corpus_topics[i], matching_cluster_centers[i]) for i in xrange(len(corpus_topics))]
    #     print potential_k, sse

    # 2	0.667763491
    # 3	0.52410108
    # 4	0.446268156
    # 5	0.381375444
    # 6	0.346909909
    # 7	0.32212836
    # 8	0.301707871
    # 9	0.282684982
    # 10	0.268554746

    elbow_k = 3
    kmeans = cluster.KMeans(n_clusters=elbow_k)
    kmeans.fit(corpus_topics)
    fitted = kmeans.predict(corpus_topics)
    matching_cluster_centers = [kmeans.cluster_centers_[fitted[i]] for i in xrange(len(fitted))]
    distances = [distance.euclidean(corpus_topics[i], matching_cluster_centers[i]) for i in xrange(len(corpus_topics))]
    pqs = [PriorityQueue() for _ in xrange(elbow_k)]

    for index in xrange(len(corpus) - 1):
        # print fitted[index], corpus[index], distances[index]
        pqs[fitted[index]].put(data_set[index], distances[i])

    for label in xrange(elbow_k):
        print label, pqs[label].get_many(10)
        print


def matching_tags(titles, tags):
    tags_to_titles = {}
    for index in xrange(len(titles)):
        index_tags = map(lambda tag: tag.strip().lower(), tags[index].split('\n'))
        index_tags = filter(lambda tag: len(tag), index_tags)
        for current_tag in index_tags:
            if not current_tag in tags_to_titles:
                tags_to_titles[current_tag] = []

            tags_to_titles[current_tag].append(titles[index])

    matches = {}
    for tag in tags_to_titles:
        titles_for_tag = tags_to_titles[tag]
        if len(titles_for_tag) > 1:
            if not tag in matches:
                matches[tag] = []

            matches[tag].extend(titles_for_tag)

    return matches


def print_tag_matches(matches):
    sorted_matches = sorted(matches.items(), key=lambda item: len(item[1]), reverse=True)
    for match in sorted_matches:
        tag, titles = match
        print '{tag} ({num}) matches:\t{titles}'.format(tag=tag, num=len(titles), titles=str(titles))


def main():
    # titles, descriptions, learning_objectives, tags = read_data(INPUT_FILE)
    # output = process_data(descriptions, learning_objectives, tags)
    # write_output(output, OUTPUT_FILE)

    titles, descriptions, learning_objectives, tags = read_data(INPUT_FILE)
    # gensim_tokenizing(descriptions)
    matches = matching_tags(titles, tags)
    print_tag_matches(matches)

if __name__ == '__main__':
    main()
