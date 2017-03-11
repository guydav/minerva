NUM_CASES = 20
NOT_POSSIBLE = 'not possible'
WARM_UP_ENERGY = 5
HEATING_ENERGY = 10


def simulate_energy(start_temp, target_temp, variance, time_minutes):
    """
    This can certainly be solved better by doing the math - if I have time
    later I'll come back to it
    """
    current_temp = start_temp - target_temp
    heater_status = False
    min_threshold_temp = -1 * variance + 1
    max_threshold_temp = variance
    energy = 0

    if current_temp > max_threshold_temp or current_temp < min_threshold_temp:
        return NOT_POSSIBLE

    for time_remaining in range(time_minutes, 0, -1):
        if heater_status is False:
            if current_temp == min_threshold_temp and time_remaining > 1:
                heater_status = True
                energy += WARM_UP_ENERGY

            current_temp -= 1

        else:
            current_temp += 1
            energy += HEATING_ENERGY

            if (current_temp == max_threshold_temp) or \
                    (time_remaining <= (current_temp - min_threshold_temp + 2)):
                heater_status = False

        # print time_remaining, heater_status, current_temp, energy

    return energy


def main():
    # while True:
    for _ in range(NUM_CASES):
        start_temp, target_temp, variance, time_minutes = \
            map(int, raw_input().split(','))

        print simulate_energy(start_temp, target_temp, variance, time_minutes)


if __name__ == '__main__':
    main()