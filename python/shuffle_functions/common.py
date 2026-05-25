import random


def uniform_shuffle(data):
    return random.sample(data, len(data))


def flatten(rows):
    result = []
    for row in rows:
        result.extend(row)
    return result
