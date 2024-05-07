import numpy as np
import textwrap

def maior_autovalor(A: np.ndarray, x0: np.ndarray):
    '''
    Funçao para determinar o autovalor dominante
    A: matriz o qual quer calclular o autovalor
    x0: chute inicial
    x: proximo interado
    y: vetor normalizado
    alfa: norma-l de y
    autovalor: autovalor dominante
    '''
    x = np.copy(x0)
    for _ in range(10):
        y = A @ x
        alfa = np.max(abs(y))
        x = (1/alfa)*y
        autovalor = np.max(y/x)
        print(f"alfa = {alfa}\nx = {x}\ny = {y}\n autovalor = {autovalor}\n\n")
    return y

def main():
    A = np.array([[1, 2], [3, 2]])
    y = np.array([1, 1])
    maior_autovalor(A, y)
    print(np.linalg.eigvals(A))

if __name__ == "__main__":
    #main()
    x = [1, 2, 3]
    print(textwrap.dedent(f'''\
    Este é um pacote Python desenvolvido para resolver sistemas lineares e não lineares utilizando uma variedade de métodos. 
    Ele inclui implementações de algoritmos como eliminação de Gauss e métodos iterativos (por exemplo, Jacobi e Gauss-Seidel), 
    bem como métodos para sistemas de equações não lineares, como o método de Newton-Raphson.
    Além disso, este pacote contém dois exemplos aplicados ao trabalho do mestrado de Modelagem Computacional da Universidade Federal Fluminense.
    {x}
    '''))
    