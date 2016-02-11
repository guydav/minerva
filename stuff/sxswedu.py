import nltk
import csv
from tqdm import tqdm
from formal.utils import PriorityQueue
import gensim
from sklearn import cluster, metrics
from scipy.spatial import distance
import itertools
import json
from graph_tool.all import *
import graph_tool.all as gt

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


class PanelSubmission(object):

    def __init__(self, line):
        self.title = line[0]
        self.track = line[1]
        self.format = line[2]
        self.understanding_level = line[3]
        self.speaker = line[4]
        self.description = clean_words(line[5])

        self.learning_objectives = []
        for col in (6, 7, 8):
            self.learning_objectives.append(clean_words(line[col]))

        self.tags = []
        for col in (9, 10, 11):
            self.tags.append(clean_words(line[col]))

        self.tags = filter(lambda tag: len(tag), map(lambda tag: tag.strip().lower(), self.tags))

        self.links = []
        for col in (12, 13, 14, 15):
            if col < len(line):
                self.links.append(clean_words(line[col]))


def read_data(path):
    # titles = []
    # descriptions = []
    # learning_objectives = []
    # tags = []

    with open(path) as data_file:
        reader = csv.reader(data_file.readlines())

        print 'Reading input:'

        return [PanelSubmission(line) for line in tqdm(reader)]

            # titles.append(clean_words(line[0]))
            # descriptions.append(clean_words(line[5]))
            # learning_objectives.append('\n'.join([clean_words(line[obj_col]) for obj_col in (6, 7, 8)]))
            # tags.append('\n'.join([clean_words(line[tag_col]) for tag_col in (9, 10, 11)]))

            # for obj_column in (6, 7, 8):
            #     learning_objectives.append(clean_words(line[obj_column]))
            # for tag_column in (9, 10, 11):
            #     tags.append(clean_words(line[tag_column]))

    # return titles, descriptions, learning_objectives, tags


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


def process_data(panel_submissions):
    # output = []
    # output.extend(process_data_set(descriptions))
    # output.extend(process_data_set(learning_objectives))
    # output.extend(process_data_set(tags))
    # output.extend(process_data_set(descriptions + learning_objectives + tags))
    output = []
    descriptions = [ps.description for ps in panel_submissions]
    learning_objectives = ['\n'.join(ps.learning_objectives for ps in panel_submissions)]
    tags = ['\n'.join(ps.tags for ps in panel_submissions)]

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


def matching_tags(panel_submissions):
    tags_to_titles = {}
    for ps in panel_submissions:
        for current_tag in ps.tags:
            if not current_tag in tags_to_titles:
                tags_to_titles[current_tag] = set()

            tags_to_titles[current_tag].add(ps.title)

    tags_to_titles = {key: tags_to_titles[key] for key in tags_to_titles if len(tags_to_titles[key]) > 1}

    titles_to_tags = {}
    for tag in tags_to_titles:
        current_titles = tags_to_titles[tag]
        num_titles = len(current_titles)

        if num_titles > 1:
            sorted_titles = sorted(current_titles)
            for first_index, second_index in itertools.combinations(xrange(num_titles), 2):
                if first_index != second_index:
                    title_tuple = (sorted_titles[first_index], sorted_titles[second_index])

                    if not title_tuple in titles_to_tags:
                        titles_to_tags[title_tuple] = set()

                    titles_to_tags[title_tuple].add(tag)

    # TODO: generate tag matching graph

    return tags_to_titles, titles_to_tags


def print_tag_matches(tags_to_titles):
    sorted_matches = sorted(tags_to_titles.items(), key=lambda item: len(item[1]), reverse=True)
    print 'Printing {num} tag => title results:'.format(num = len(sorted_matches))
    for match in sorted_matches:
        tag, titles = match
        print '{tag} ({num}) matches:\t{titles}'.format(tag=tag, num=len(titles), titles=str(titles))


def print_clicks(titles_to_tags, min_length=2):
    filtered_matches = filter(lambda item: len(item[1]) >= min_length, titles_to_tags.items())
    sorted_matches = sorted(filtered_matches, key=lambda item: len(item[1]), reverse=True)

    print '\nPrinting {num} match => matching tags results:'.format(num=len(sorted_matches))
    tagset_to_matches = {}
    for match in sorted_matches:
        title_pair, tags = match
        first_title, second_title = title_pair
        print '{first} <=> {second} matched on: {tags}'.format(first=first_title, second=second_title, tags=str(tags))

        tags_tuple = tuple(tags)
        if tags_tuple not in tagset_to_matches:
            tagset_to_matches[tags_tuple] = set()

        tagset_to_matches[tags_tuple].update(title_pair)

    sorted_tagsets = sorted(tagset_to_matches.items(), key=lambda item: len(item[0]), reverse=True)
    print '\nPrinting {num} tag set => matching titles results:'.format(num=len(sorted_tagsets))
    for tags, titles in sorted_tagsets:
        print '{tags} => {titles}'.format(tags=tags, titles=tuple(titles))

    return tagset_to_matches


def write_network_json(panel_submissions, output_path=None):
    submissions = []
    tag_dict = {}
    link_dict = {}

    for ps in panel_submissions:
        submission_id = hash(ps)
        submissions.append({
            'id': submission_id,
            'title': ps.title,
            'speaker': ps.speaker,
            'tags': [hash(tag) for tag in ps.tags]
        })

        for tag in ps.tags:
            if not tag in tag_dict:
                tag_dict[tag] = []

            tag_dict[tag].append(submission_id)

        for first_tag, second_tag in itertools.combinations(ps.tags, 2):
            first_tag_id = hash(first_tag)
            second_tag_id = hash(second_tag)

            if first_tag_id <= second_tag_id:
                key = (first_tag_id, second_tag_id)
            else:
                key = (second_tag_id, first_tag_id)

            if not key in link_dict:
                link_dict[key] = []

            link_dict[key].append(submission_id)

    tags = [{'name': tag, 'id': hash(tag), 'submissions': sorted(tag_dict[tag])} for tag in tag_dict]
    links = [{'source': link[0], 'target': link[1], 'submissions': sorted(link_dict[link])} for link in link_dict]

    output = {'tags': tags, 'links': links, 'submissions': submissions}

    if output_path:
        with open(output_path, 'w') as output_file:
            json.dump(output, output_file, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))

    return tags, links, submissions


def draw_graph(tags, links):
    graph = gt.Graph(directed=False)
    vertex_names = graph.new_vertex_property('string')
    vertex_sizes = graph.new_vertex_property('double')
    edge_width = graph.new_edge_property('double')

    tag_id_to_vertex = {}

    for tag in tags:
        vertex = graph.add_vertex()
        vertex_names[vertex] = tag['name']
        vertex_sizes[vertex] = len(tag['submissions'])
        tag_id_to_vertex[tag['id']] = vertex

    for link in links:
        edge = graph.add_edge(tag_id_to_vertex[link['source']], tag_id_to_vertex[link['target']])
        edge_width[edge] = len(link['submissions'])

    gt.graph_draw(g, vertex_text=vertex_names, vertex_size=vertex_sizes, edge_pen_width=edge_width)

def main():
    # panel_submissions = read_data(INPUT_FILE)
    # output = process_data(panel_submissions)
    # write_output(output, OUTPUT_FILE)

    panel_submissions = read_data(INPUT_FILE)
    # gensim_tokenizing([ps.description for ps in panel_submissions])
    # tags_to_titles, titles_to_tags = matching_tags(panel_submissions)
    # print_tag_matches(tags_to_titles)
    # print_clicks(titles_to_tags)
    tags, links, submissions = write_network_json(panel_submissions, 'tag_network.json')
    # draw_graph(tags, links)

if __name__ == '__main__':
    main()
