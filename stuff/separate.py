

def main():
    sets = []

    with open('/Users/guydavidson/Downloads/retail_25k.dat') as f:
        for line in f.readlines():
            group = set(map(int, line.strip().split(' ')))

            matching_sets = filter(lambda s: not s.isdisjoint(group), sets)

            if not matching_sets:
                sets.append(group)

            else:
                [sets.remove(ms) for ms in matching_sets]
                start_set = matching_sets[0]
                for additional_set in matching_sets[1:]:
                    start_set.update(additional_set)

                start_set.update(group)
                sets.append(start_set)

    for final_set in sets:
        print len(final_set)


if __name__ == '__main__':
    main()