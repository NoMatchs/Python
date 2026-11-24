def kahan_sum(values):
    total = 0.0
    c = 0.0
    for x in values:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def kahan_sum_iter(iterable, start=0.0):
    total = float(start)
    c = 0.0
    for x in iterable:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


if __name__ == "__main__":
    data = [1.0] + [1e-17] * 10

    print("data->", data)
    print("Built-in sum->", sum(data))
    print("Kahan sum->", kahan_sum(data))
    print("kahan sum iter->", kahan_sum_iter(data))
