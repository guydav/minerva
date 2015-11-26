import csv
import numpy

# -- Constants -- #
DATA_FILE = '1915_2015_team_data_raw.csv'



# -- Helper Functions -- #
def read_data(path):
    data = []

    with open(path, 'r') as data_file:
        reader = csv.reader(data_file)
        headers = reader.next()

        for row in reader:
            data.append({headers[i]: row[i] for i in xrange(len(headers))})

    return data

# -- Main -- #

def main():
    data = read_data(DATA_FILE)


if __name__ == '__main__':
    main()
