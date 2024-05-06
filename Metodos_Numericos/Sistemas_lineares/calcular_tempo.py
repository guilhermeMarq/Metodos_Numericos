import time

def tempo_execuçao(funçao):

    n_repetiçoes = 100
    inicio = time.time()

    for _ in range(n_repetiçoes):
        funçao()

    final = time.time()
    tempo_medio = 1000*(final - inicio)/n_repetiçoes

    return tempo_medio
