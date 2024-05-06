from src import Sistemas_lineares
from src import tempo_execuçao
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

def minimos_quadrados(funcao: str, *, n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    x = sp.symbols("x")

    #Construindo a matriz A (integral x^(j+i-2))
    A = np.empty((n, n), dtype=object)

    for i in range(1, n+1):
        for j in range(1, n+1):
            A[i-1][j-1] = sp.integrate(x**(i+j-2), (x, a, b))

    #Construindo o vetor B
    F = sp.sympify(funcao)
    B = np.empty(n, dtype=object)
    for i in range(1, n+1):
        B[i-1] = sp.integrate(F*x**(i-1), (x, a, b))

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    return A, B

def grafico(F, P):

    x = sp.symbols('x')
    F = sp.simplify(F)
    P = sp.simplify(P)
    
    # Converta as funções sympy para funções numpy para poder plotar
    F_numpy = sp.lambdify(x, F, 'numpy')
    P_numpy = sp.lambdify(x, P, 'numpy')  # Corrigido de G_numpy para P_numpy
    
    # Crie um intervalo de valores de x para plotar
    x_values = np.linspace(-5, 5, 400)

    # Calcule os valores de y correspondentes para cada função usando as funções numpy
    y_values_F = F_numpy(x_values)
    y_values_P = P_numpy(x_values)  # Corrigido de y_values_G para y_values_P
    
    # Plotar o gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values_F, label='F(x)')  # Adicione um rótulo se desejar
    plt.plot(x_values, y_values_P, label='P(x)')  # Adicione um rótulo para a segunda função
    plt.xlabel('x')
    #plt.ylabel('y')
    plt.title('Gráfico das funções F(x) e P(x)')
    plt.grid(True)
    plt.legend()
    #plt.xlim(0,1)
    plt.ylim(-2, 8)
    plt.show()

    return

def erro_absoluto(F, P):
    x = sp.symbols('x')
    F = sp.simplify(F)
    P = sp.simplify(P)

    a = 0
    b = 1

    erro = 1/(b-a)*sp.integrate(abs(F-P), (x, a, b))

    print(f"erro = {erro}")

def main():
    funcao = "x**3 - x**2 + x + 1"
    grau = 3

    A, b = minimos_quadrados(funcao, n=grau, a=0, b=1)
    meusistema = Sistemas_lineares.SistemaLinearSolver(A, b)

    meusistema.jacobi(relatorio=True)
    meusistema.seidel(relatorio=True)
    meusistema.relaxamento(relatorio=True)
    meusistema.gradiente_conjugado(relatorio=True)
    meusistema.gradiente_conjugado_quadrado(relatorio=True)
    meusistema.eliminaçao_gauss(relatorio=True)

    '''tempo_medio = {"Eliminaçao de gaus":  tempo_execuçao(meusistema.eliminaçao_gauss),
                    "Jacobi":             tempo_execuçao(meusistema.jacobi),
                    "Seidel":             tempo_execuçao(meusistema.seidel),
                    "Relaxamento":        tempo_execuçao(meusistema.relaxamento),
                    "Gradiente_conjugado":tempo_execuçao(meusistema.gradiente_conjugado),
                    "Conjugado quadrado": tempo_execuçao(meusistema.gradiente_conjugado_quadrado)}
    print(tempo_medio)
'''
    #Construir o grafico
    P = ""
    for i in range(grau):
        P += f" + {meusistema.x[i]}*x**{i}"

    grafico(funcao, P)
    #erro_absoluto(funcao, P)


if __name__ == "__main__":
    main()