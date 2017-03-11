NUM_RINGS = 3


def decrypt_enigma(rings, encrypted_message):
    output = []

    for c_enc in encrypted_message:
        c_dec = (int(c_enc, 16) - sum(map(lambda r: int(r[0], 16), rings))) % 16
        output.append(hex(c_dec)[-1])

        _, max_index = sorted([(rings[i][0], -1 * i) for i in
                               range(len(rings))], reverse=True)[0]
        max_index *= -1
        max_ring = rings[max_index]
        rings[max_index] = max_ring[1:] + max_ring[0]

    return ''.join(output).decode('hex')


def main():
    rings = [raw_input() for _ in range(NUM_RINGS)]
    encrypted_message = raw_input()
    print decrypt_enigma(rings, encrypted_message)


if __name__ == '__main__':
    main()