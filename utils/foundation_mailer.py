__author__ = 'guydavidson'

import json
import gspread
import gspread.exceptions
from oauth2client.client import SignedJwtAssertionCredentials
import csv
import re
import datetime
import smtplib
import getpass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import jinja2
import gdata.gauth
import gdata.contacts.client
import atom
import httplib2

START_DATE = datetime.date(2015, 8, 31)
# USERS_FILE = 'users.txt'
USERS_FILE = 'Emails for Foundation Mailer - Sheet1.csv'
CREDENTIALS_FILE = '/Users/guydavidson/virtualenvs/dev/data/guy_minerva_updated_173754e761cf.json'
SPREADSHEET_NAME = 'SBCC'
NAME_PATTERN = '.*{name}.*'
SESSION_ROW = 6
HEADERS_ROW = 7

INDICES = {0: [(1, 8), (8, 16)],
           1: [(1, 4), (4, 7), (7, 11)],
           2: [(1, 3), (3, 6), (6, 14), (14, 20)],
           3: [(1, 11), (11, 12), (12, 19)],
           4: [(1, 8), (8, 9), (9, 12)]}
GMAIL_SMTP = 'smtp.gmail.com:587'
MY_EMAIL = 'guy@minerva.kgi.edu'
TEMPLATE_FILE = 'template.html'
PASSWORD_PROMPT = 'Please enter your password for {email}: '.format(email=MY_EMAIL)

smtp_server = None

def auth():
    with open(CREDENTIALS_FILE) as credentials_file:
        json_key = json.load(credentials_file)
        scope = ['https://spreadsheets.google.com/feeds']

        credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'], scope)

        return gspread.authorize(credentials)

def read_users():
    with open(USERS_FILE) as users:
        reader = csv.reader(users)
        return [line for line in reader][1:]

def get_worksheet_index():
    today = datetime.date.today()
    # today = datetime.date(2015, 9, 4)
    delta = today - START_DATE
    return delta.days
    # return delta.days + 1

def read_and_email(worksheet, name, email, day_index):
    cell = None

    try:
        compiled_name_pattern = re.compile(NAME_PATTERN.format(name=name.strip()))
        cell = worksheet.find(compiled_name_pattern)

    except gspread.exceptions.CellNotFound:
        print "Couldn't find cell for {name}, aborting".format(name=name)
        return

    user_row = cell.row

    prepare_and_send_email(worksheet.row_values(SESSION_ROW), worksheet.row_values(HEADERS_ROW),
                           worksheet.row_values(user_row), day_index, name, email)

class Table(object):
    def __init__(self, session, headers, data):
        self.session = session
        self.headers = headers
        self.data = data

def prepare_and_send_email(sessions, headers, user_values, day_index, name, email):
    sessions = [sess for sess in sessions[1:] if sess]

    global smtp_server
    if not smtp_server:
        smtp_server = smtplib.SMTP(GMAIL_SMTP)
        smtp_server.starttls()
        smtp_server.login(MY_EMAIL, getpass.getpass(PASSWORD_PROMPT))

    msg = MIMEMultipart('alternative')
    msg['Subject'] = "Foundation Mailer for {date}".format(date=datetime.date.today().strftime('%m/%d'))
    # msg['Subject'] = "Foundation Mailer for {date}".format(date='09/03')
    msg['From'] = MY_EMAIL
    msg['To'] = email

    tables = []
    indices = INDICES[day_index]
    for i in xrange(len(indices)):
        start, end = INDICES[day_index][i]
        table = Table(sessions[i], headers[start:end], user_values[start:end])
        tables.append(table)

    html = None
    with open(TEMPLATE_FILE) as template_file:
        template = jinja2.Template(template_file.read())
        html = template.render(name = name, tables = tables)

    part = MIMEText(html, 'html')
    msg.attach(part)

    smtp_server.sendmail(MY_EMAIL, email, msg.as_string())

    print 'Successfully emailed {email}'.format(email=email)


# def name_to_email(name):
#     client = None
#     with open(CREDENTIALS_FILE) as credentials_file:
#         json_key = json.load(credentials_file)
#         scope = ['https://www.google.com/m8/feeds']
#
#         auth_token = gdata.gauth.OAuth2Token(client_id=json_key['client_id'], client_secret=json_key['private_key'],
#                                         scope=scope, user_agent='GData-Version: 3.0')
#
#         redirect_url = auth_token.generate_authorize_url(
#             'https://www.google.com/m8/feeds/contacts/{userEmail}/full'.format(userEmail=MY_EMAIL))
#
#         return redirect_url
#
#         # url = atom.http_core.ParseUri(redirect_url)
#         # access_token = auth_token.get_access_token(url.query)
#
#         print access_token
#
#         client = gdata.contacts.client.ContactsClient(source='My Project', auth_token=token)
#
#         #print client.GetContacts()
#
#     return client


def main():
    day_index = get_worksheet_index()

    client = auth()
    spreadsheet = client.open(SPREADSHEET_NAME)
    worksheet = spreadsheet.get_worksheet(day_index)

    users = read_users()
    for name, email in users:
        read_and_email(worksheet, name, email, day_index)

if __name__ == '__main__':
    main()