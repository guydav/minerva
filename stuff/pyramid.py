#! /usr/local/bin/python


OUR_NAMES = ['Guy', 'Eli', 'Margot', 'Guilherme', ]


def get_names():
    names = []
    name = raw_input("Please enter one of your first names, or nothing if you're done: ")

    while name:
        names.append(name.strip())
        name = raw_input("Please enter one of your first names, or nothing if you're done: ")

    return names


def prepare_name(name):
    return ' '.join(name.upper())


def print_names(names):
    lines = []
    names.sort(key=str.__len__)
    lines.append(prepare_name(names[0]))

    for name in names[1:] + OUR_NAMES:
        prepared_name = prepare_name(name)
        ratio = len(lines[-1]) // len(prepared_name) + 1
        lines.append(' | '.join([prepared_name] * ratio))

    lines.insert(0, prepare_name('WELCOME TO PYRAMID'))
    lines.insert(1, '')

    max_length = len(lines[-1])
    format_string = '{:^' + str(max_length) + '}'
    lines = [format_string.format(line) for line in lines]

    print
    for line in lines:
        print line
    print


def main():
    names = get_names()
    print_names(names)


if __name__ == '__main__':
    main()