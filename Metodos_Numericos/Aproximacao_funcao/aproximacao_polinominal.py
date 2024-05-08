import numpy as np
import sympy as sp


def interpolaçao(pontos: list):
    '''
    pontos: Pontos de interpoção no formato [(x1, y1), (x2, y2), ...]
    grau: grau do polinomio de aproximaçao desejado
    '''
    n = len(pontos)
    x = sp.symbols("x")
    l = np.ones(n, dtype=object)

    for i in range(n):
        for j in range(n):
            if i != j:
                l[i] *= (x - pontos[j][0])/(pontos[i][0] - pontos[j][0])

    P = 0
    for i in range(n):
        P += pontos[i][1]*l[i]
    
    return P.expand()


if __name__ == "__main__":

    pontos = [[100, 10], [121, 11], [144, 12]]
    print(interpolaçao(pontos))