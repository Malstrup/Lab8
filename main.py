import numpy as np
import matplotlib.pyplot as plt


def k1(h, t, v, kounter):
    return h * fs(t, v, kounter)


def k2(h, t, v, kounter, k1):
    return h * fs(t + h / 2, v + k1 / 2, kounter)


def k3(h, t, v, kounter, k1, k2):
    return h * fs(t + h, v + 2 * k2 - k1, kounter)


def next(h, t, v, kounter):
    k11 = k1(h, t, v, kounter)
    k21 = k2(h, t, v, kounter, k11)
    k31 = k3(h, t, v, kounter, k11, k21)
    return v + 1 / 6 * (k11 + 4 * k21 + k31)


def dlt(v, v1):
    return np.linalg.norm(np.array(v) - np.array(v1), ord=2) / (pow(2, 3) - 1)


def method(t, T, h, v, kounter, eps):
    v1 = v
    v2 = v
    x = []
    y = []
    while t < T + h / 2:
        v_0 = v
        print("{:13.6f}".format(t), "{:12.6f}".format(h), "{:15.5e}".format(dlt(v, v2)), "{:12d}".format(kounter[0]),
              end=' ')
        for vi in v_0:
            print("{:12.6f}".format(vi), end=' ')
        print()
        v = next(h, t, v_0, kounter)
        v1 = next(h / 2, t, v_0, kounter)
        v2 = next(h / 2, t + h / 2, v1, kounter)
        while dlt(v, v2) > eps:
            h /= 2
            v = next(h, t, v_0, kounter)
            v1 = next(h / 2, t, v_0, kounter)
            v2 = next(h / 2, t + h / 2, v1, kounter)
        while dlt(v, v2) < eps / 64:
            h *= 2
            v = next(h, t, v_0, kounter)
            v1 = next(h / 2, t, v_0, kounter)
            v2 = next(h / 2, t + h / 2, v1, kounter)
        t += h
        x.append(t)
        y.append(h)
    print()
    x.pop()
    y.pop()
    return x, y, min(y), len(x)


def answer(t, T, h, v, kounter):
    epses = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    k = 0
    minhes = []
    numstepses = []
    fig, ax = plt.subplots(2, 3)
    fig.tight_layout()
    for item in epses:
        ax[k // 3, k % 3].set_title('Точность равна ' + str(item))
        ax[k // 3, k % 3].set_xlabel('Отрезок')
        ax[k // 3, k % 3].set_ylabel('Шаг по отрезку')
        print('Точность равна ', item)
        x, y, minh, numsteps = method(t, T, h, v, kounter, item)
        minhes.append(minh)
        numstepses.append(numsteps)
        ax[k // 3, k % 3].scatter(x, y)
        k += 1
    plt.figure()
    plt.semilogx(epses, minhes)
    plt.xlabel('Точность')
    plt.ylabel('Минимальный шаг')
    plt.title('Зависимость минимального шага от точности')
    plt.figure()
    plt.semilogx(epses, numstepses)
    plt.xlabel('Точность')
    plt.ylabel('Число шагов')
    plt.title('Зависимость числа шагов от точности')
    plt.show()


if __name__ == '__main__':
    func = []
    t_0 = float(input())
    T = float(input())
    h_0 = float(input())
    N_x = int(input())
    eps = float(input())
    n = int(input())
    for i in range(n + 3):
        func.append(input())
    b = ''
    for s in func:
        b = b + s + '\n'
    v_0 = input().split()
    for i in range(n):
        v_0[i] = float(v_0[i])
    t = t_0
    v = v_0
    v1 = v_0
    v2 = v_0
    h = h_0
    kounter = [0]
    exec(b)
    answer(t, T, h, v, kounter)

