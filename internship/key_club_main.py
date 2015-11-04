__author__ = 'guydavidson'

import key_club

### Main

def main():
    records = key_club.load_csv(key_club.FULL_FILE)
    key_club.google_records(records)
    # find_emails(records)
    #
    # for record in records:
    #     print record

if __name__ == '__main__':
    main()