import random


def fisher_yates(data, B=None):
    result = list(data)
    for i in range(len(result) - 1, 0, -1):
        j = random.randint(0, i)
        result[i], result[j] = result[j], result[i]
    return result
