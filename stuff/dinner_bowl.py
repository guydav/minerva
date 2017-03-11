ONE_HOUR_IN_MINUTES = 60
WALKING_TIME = 10
FOUR_HOURS = 4 * ONE_HOUR_IN_MINUTES
MAXIMUM_BOWLING_TIME = FOUR_HOURS - 10


def solve_week(group_size):
    times = [sum(map(int, raw_input().split(' '))) for _ in range(group_size)]
    dinner_time = max(times)
    return max(230 - dinner_time, 0)


def main():
    num_cases = int(raw_input())

    for _ in range(num_cases):
        group_size = int(raw_input())
        print solve_week(group_size)

    print


if __name__ == '__main__':
    main()