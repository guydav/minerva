import datetime
import csv
import google
import requests
import re
import redis
import rq
import peewee
import sortedcontainers

__author__ = 'guydavidson'

QUEUE = rq.Queue(connection=redis.Redis(), default_timeout=300)
TEST_DB_PATH = '/Users/guydavidson/PycharmProjects/Minerva/internship/key_club_alabama.sqlite'
FULL_DB_PATH = '/Users/guydavidson/PycharmProjects/Minerva/internship/key_club_all.sqlite'
DB = peewee.SqliteDatabase(FULL_DB_PATH)

# -- Constants -- #

DEBUG = True
TEST_FILE = '/Users/guydavidson/PycharmProjects/Minerva/internship/key_club_alabama.csv'
FULL_FILE = '/Users/guydavidson/PycharmProjects/Minerva/internship/key_club.csv'
IGNORE_DOMAINS = ('kiwanis.org', 'ussearch.com', 'classfinders.com', 'www.ahsaa.com', 'http://vermilionsheriff.net/',
                  'http://www.scribd.com/')
IGNORE_SUFFIXES = ('.pdf',)
HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 ' +
                              '(KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36'}
EMAIL_REGEXP = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
EMAIL_PATTERN = re.compile(EMAIL_REGEXP)
MAX_PAGE_SIZE = 10 ** 6

QUERIES = ('{name} {supervisor}', '{name} {supervisor} key club', '{name} {supervisor} contact email')


# --  Classes -- #


class School(peewee.Model):
    id = peewee.PrimaryKeyField()
    name = peewee.TextField()
    state = peewee.TextField()
    supervisor = peewee.TextField()
    club_code = peewee.TextField()

    class Meta:
        database = DB

    def build_queries(self):
        if not self.supervisor or len(str(self.supervisor)) == 0:
            return None

        return [query.format(name=self.name, supervisor=self.supervisor) for query in QUERIES]

    def __repr__(self):
        return '{supervisor} - {name} ({state})'.format(supervisor=self.supervisor, name=self.name, state=self.state)


class Url(peewee.Model):
    id = peewee.PrimaryKeyField()
    school = peewee.ForeignKeyField(School, related_name='urls')
    url = peewee.TextField()
    checked = peewee.IntegerField()

    class Meta:
        database = DB

    def __repr__(self):
        return '{url} checked: {checked}'.format(url=self.url, checked=self.checked)


class Email(peewee.Model):
    id = peewee.PrimaryKeyField()
    url = peewee.ForeignKeyField(Url, related_name='emails')
    email = peewee.TextField()
    priority = peewee.IntegerField()

    class Meta:
        database = DB

    def __repr__(self):
        return '{email} {priority}'.format(email=self.email, priority=(self.priority and '!!' or ''))


class PriorityQueue(object):
    def __init__(self):
        self.priority_to_elements = sortedcontainers.SortedDict()
        self.element_to_priority = sortedcontainers.SortedDict()

    def put(self, element, priority):
        # Assuming a lower priority is better
        if (element in self.element_to_priority) and (priority > self.element_to_priority[element]):
            return

        self.element_to_priority[element] = priority

        if priority not in self.priority_to_elements:
            self.priority_to_elements[priority] = []

        self.priority_to_elements[priority].append(element)

    def get(self, min=True):
        full_get = self.get_with_priority(min)

        if full_get:
            return full_get[0]

        return None

    def get_with_priority(self, min=True):
        if not self.priority_to_elements:
            return None

        keys = self.priority_to_elements.keys()
        if min:
            key = keys[0]

        else:
            key = keys[len(keys) - 1]

        priority = self.priority_to_elements[key]

        element = priority.pop()

        if not priority:
            del self.priority_to_elements[key]

        return element, key

    def __len__(self):
        return len(self.priority_to_elements)


# -- Functions -- #


def debug(message):
    if DEBUG:
        print '{time}:\t{message}'.format(time=str(datetime.datetime.now()), message=message)


def load_csv(path):
    debug('Loading file {path}'.format(path=path))
    records = []
    with open(path) as test_file:
        reader = csv.reader(test_file)

        for line in reader:
            name, alt_name, state, supervisor, club_code = line[:5]

            if not name:
                name = alt_name

            school, created = School.get_or_create(name=name, state=state, supervisor=supervisor, club_code=club_code)
            records.append(school)

    return records


def or_join_list(boolean_list):
    return reduce(lambda x, y: x or y, boolean_list, False)


def filter_results(results):
    return [result for result in results if not
            or_join_list([domain.lower() in result.lower() for domain in IGNORE_DOMAINS] +
                         [result.lower().endswith(suffix) for suffix in IGNORE_DOMAINS])]


def google_records(schools):
    # Starting with testing one
    for school in schools:
        try:
            queries = school.build_queries()

            if queries:
                filtered_results = set()

                for query in queries:
                    debug('Querying google for {query}'.format(query=query))

                    results = google.search(query, stop=10)
                    filtered_results = filtered_results.union(filter_results(results))

                for url in filtered_results:
                    Url.get_or_create(school=school, url=url, checked=0)

                for url in school.urls:
                    QUEUE.enqueue(find_emails, url.id)

        except BaseException, e:
            debug('Ran into an exception {exc} with message {msg}'.format(exc=str(e), msg=e.message))


LAST_NAME_WEIGHT = 8
K12_WEIGHT = 4
FIRST_NAME_WEIGHT = 2
FIRST_INITIAL_WEIGHT = 1
LAST_INITIAL_WEIGHT = 1

K12 = 'k12'


def email_heuristic(first_name, last_name, email):
    address_score_dict = {last_name: LAST_NAME_WEIGHT, first_name: FIRST_NAME_WEIGHT,
                  last_name[0]: LAST_INITIAL_WEIGHT, first_name[0]: FIRST_INITIAL_WEIGHT}

    domain_score_dict = {K12: K12_WEIGHT}

    lower_email = email.lower()
    address, domain = lower_email.split('@')

    # score = 0
    # for key in score_dict:
    #     if key.lower() in lower_email:
    #         score += score_dict[key]

    return sum([address_score_dict[key] if key.lower() in address else 0 for key in address_score_dict]) + \
        sum([domain_score_dict[key] if key.lower() in domain else 0 for key in domain_score_dict])


def find_emails(url_id):
    url = Url.get(id=url_id)
    response = requests.get(url.url, headers=HTTP_HEADERS)
    size = len(response.text)

    if size < MAX_PAGE_SIZE:
        debug('Searching in {url} size is {size}'.format(url=url.url, size=size))
        emails = set(EMAIL_PATTERN.findall(response.text))

        if emails:
            debug('In {url}, found {count} emails'.format(url=url.url, count=len(emails)))
            supervisor = url.school.supervisor

            for email in emails:
                priority = 0

                if supervisor:
                    name_split = [s for s in supervisor.split() if len(s) > 1]

                    if not name_split:
                        continue

                    if len(name_split) == 1:
                        name_split.append('')

                    first_name = name_split[0]
                    last_name = name_split[1]

                    priority = email_heuristic(first_name, last_name, email)

                email_record, created = Email.get_or_create(url=url, email=email, priority=priority)
                debug('Created email {email}'.format(email=email_record))

    else:
        debug('Skipping {url} because size is {size}'.format(url=url, size=size))

    url.checked = 1
    url.save()


HEADERS = ('Organization', 'First', 'Last', 'School', 'Title', 'State', 'Key Club Code', 'Top Email', 'Confidence',
           'Top Ten Emails')
KEY_CLUB = 'Key Club'
TITLE = 'Advisor'
MAXIMUM_EMAILS = 100
EMAIL_ONLY_SUFFIX = '_emails_only.csv'


def create_tables():
    DB.connect()
    DB.create_tables([School, Url, Email])


def update_priorities():
    DB.connect()
    changed_count = 0
    changed_schools = []

    for school in School.select():
        supervisor = school.supervisor
        changed = False

        if supervisor:
            name_split = [s for s in supervisor.split() if len(s) > 1]

            if not name_split:
                continue

            if len(name_split) == 1:
                name_split.append('')

            first_name = name_split[0]
            last_name = name_split[1]

            for url in school.urls:
                for email in url.emails:
                    priority = email_heuristic(first_name, last_name, email.email)
                    if priority != email.priority:
                        if not changed:
                            changed_count += 1

                        changed = True

                        email.priority = priority
                        email.save()

        if changed:
            changed_schools.append(school)
            print 'Updating priority for {school}'.format(school=school)

    print 'In total, updated {count} schools'.format(count=changed_count)

    write_schools('schools_with_changed_priorities.csv', changed_schools)

    DB.commit()




def write_schools(output_name, schools=None):
    email_rows = []
    all_rows = []

    if not schools:
        schools = School.select()

    for school in schools:
        emails = [[email for email in url.emails] for url in school.urls]
        emails = reduce(lambda x, y: x + y, emails, [])

        email_set = set()

        email_pq = PriorityQueue()
        for email in emails:
            if email.email in email_set:
                continue

            if not email.priority:
                email.priority = 0

            email_pq.put(email.email, email.priority)
            email_set.add(email.email)

        first = ''
        last = ''

        if school.supervisor:
            split = school.supervisor.split()
            first = split[0]

            if len(split) > 1:
                last = ' '.join(split[1:])

        row = [KEY_CLUB, first, last, school.name, TITLE, school.state, school.club_code]

        if email_set:
            top_email, priority = email_pq.get_with_priority(min=False)
            row.append(top_email)
            row.append(priority)
            top_ten_emails = []

            i = 0
            while i < 10 and len(email_pq):
                top_ten_emails.append(email_pq.get(min=False))
                i += 1

            row.append(', '.join(top_ten_emails))
            email_rows.append(row)

        else:
            row.append('') # First Email
            row.append('') # Top Email Priority ("Confidence")
            row.append('') # Top Ten Emails

        all_rows.append(row)

    outputs = [(output_name, all_rows), (output_name + EMAIL_ONLY_SUFFIX, email_rows)]

    for output in outputs:
        file_name, rows = output
        with open(file_name, 'w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(HEADERS)
            writer.writerows(rows)
