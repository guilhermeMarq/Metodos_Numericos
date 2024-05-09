# Biblioteca para manipulação de matrizes e vetores
import numpy as np

# Biblioteca para manipulação e resoluçao de expressões matemáticas
import math

def derivada_aproximacao(pontos: list[float], valor_ponto: float, ordem_derivada: int) -> float:
    """
    Calcula aproximadamente a derivada de ordem n de um conjunto de pontos em um valor específico.
    
    Argumentos:
        pontos (list): Uma lista que representa os pontos no conjunto de dados. ex: [[x1, y1], [x2, y2], ...]
        valor_ponto (float): O valor em que se deseja calcular a derivada.
        ordem (int): A ordem da derivada a ser calculada.
        
    Return:
        Df (float): A derivada de ordem n aproximada no valor especificado.
    """
    n = len(pontos)
    h = np.zeros(n, dtype=float)
    for i in range(n):
        h[i] = pontos[i][0] - valor_ponto

    #serie de Taylor
    F = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            F[i][j] += h[i]**j/math.factorial(j)
    F = F.transpose()

    b = np.zeros(n, dtype=float)
    b[ordem_derivada] = 1

    #Resoluçao do sistema linear
    x = np.linalg.solve(F, b)
    
    #Calculo da derivada de ordem n
    Df = 0
    for i in range(n):
        Df += pontos[i][1]*x[i]

    return Df

def integral_aproximacao():
    "Em construçao..."
    return
