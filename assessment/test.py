
import tabulate
headers = ('A', 'B', 'C', 'D', '(A & B & C) -> D', '(A & B) -> (C -> D)')

results = []
for a in (True, False):
    for b in (True, False):
        for c in (True, False):
            for d in (True, False):
                results.append((a, b, c, d, not (a and b and c and not d), not ((a and b) and (c and not d))))

print tabulate.tabulate(results, headers)