import elasticsearch_dsl as dsl
from elasticsearch_dsl.connections import connections
import random
import sys
import base64
import argparse


CLICK_INDEX = 'click'
IMPRESSION_INDEX = 'impression'
JANUARY_FIRST_TIMESTAMP = 1483228800000
APRIL_FIRST_TIMESTAMP = 1491004800000
SOURCE_OPTIONS = ('mobile', 'desktop')
URL_FORMAT = 'https://www.youtube.com/watch?v={id}'
CLICK_PROBABILITY_GIVEN_TEST_GROUP = {False: 0.2, True: 0.3}
TIME_STEP = (100, 10000)
PROBABILITY_SESSION_CONTINUES = 0.75


class Click(dsl.DocType):
    """
    A model class for a click - describes the different fields in the click
    index
    """
    class Meta:
        index = 'click'

    session_id = dsl.Long()
    source = dsl.Text()
    test_group = dsl.Boolean()
    timestamp = dsl.Date()
    url = dsl.Text()
    video_id = dsl.Long()


class Impression(dsl.DocType):
    """
    A model class for an impression - describes the different fields in the
    impression index
    """
    class Meta:
        index = 'impression'

    session_id = dsl.Long()
    source = dsl.Text()
    test_group = dsl.Boolean()
    timestamp = dsl.Date()
    url = dsl.Text()
    video_id = dsl.Long()


def generate_random_session():
    """
    Generate data into the elastic search, using a probabilistic model.
    Time stamp, video id, source, test group, and session id are generated
    randomly, and the URL is a function of the video id.

    Afterwards, continue a session with a fixed probability (currently hardcoded
    to 0.75), and as long as the session continues, generate a click with a
    probability based on being in the test group or not (currently 0.3 if yes,
    0.2 if not).
    :return: None; impressions and clicks form session saved to ES
    """
    timestamp = random.randint(JANUARY_FIRST_TIMESTAMP, APRIL_FIRST_TIMESTAMP)
    video_id = random.randint(1, sys.maxint)
    source = random.choice(SOURCE_OPTIONS)
    test_group = bool(random.randint(0, 1))
    session_id = random.randint(1, sys.maxint)
    url = URL_FORMAT.format(id=base64.b64encode(str(video_id)))

    session_continues = True

    while session_continues:
        timestamp += random.randint(*TIME_STEP)
        impression = Impression(timestamp=timestamp, video_id=video_id,
                                source=source, test_group=test_group,
                                session_id=session_id, url=url)
        impression.save()

        if random.random() < CLICK_PROBABILITY_GIVEN_TEST_GROUP[test_group]:
            timestamp += random.randint(*TIME_STEP)
            click = Click(timestamp=timestamp, video_id=video_id,
                          source=source, test_group=test_group,
                          session_id=session_id, url=url)
            click.save()

        session_continues = random.random() < PROBABILITY_SESSION_CONTINUES


def calculate_naive_ctr():
    """
    Naive click-through rate calculation. Divide the total number of clicks by
    the total number of impressions.
    :return: The naive CTR as a float
    """
    impression_search = Impression.search()
    click_search = Click.search()
    return float(click_search.count()) / impression_search.count()


def calculate_session_ctr():
    """
    A less-naive click-through rate calculation, counting impressions for each
    session together, and clicks for each session together. In other words,
    counting the number of unique sessions in the clicks index, and dividing
    by the number of unique sessions in the impressions index
    :return: The by-session CTR as a float
    """
    impression_sessions = _count_unique_sessions(Impression)
    click_sessions = _count_unique_sessions(Click)
    return float(click_sessions) / impression_sessions


def _count_unique_sessions(model_class):
    """
    Helper method to count unique sessions for a given model class, using ES's
    cardinality metric for the size of a set
    :param model_class: The model class to count - must have a session_id member
    :return: The number of unique sessions, as estimated by ES's cardinality
        metric
    """
    search = model_class.search().extra(size=0)
    search.aggs.metric('unique_sessions', 'cardinality', field='session_id')
    results = search.execute()
    return results.aggs.unique_sessions.value


def main():
    connections.create_connection(hosts=['localhost'])

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate_data', required=False,
                       type=int, help='How many data points to generate')
    group.add_argument('-c', '--ctr', required=False,
                       action='store_true', help='Calculate CTR naively')
    group.add_argument('-s', '--session_ctr', required=False,
                       action='store_true', help='Calculate CTR per session')

    args = parser.parse_args()

    if args.generate_data:
        for i in range(args.generate_data):
            print i
            generate_random_session()

    elif args.ctr:
        print 'The naive CTR is {ctr:.3f}%'.format(
            ctr=calculate_naive_ctr() * 100)

    else:  # args.session_ctr
        print 'The by-session CTR is {ctr:.3f}%'.format(
            ctr=calculate_session_ctr() * 100)
        calculate_session_ctr()


if __name__ == '__main__':
    main()
