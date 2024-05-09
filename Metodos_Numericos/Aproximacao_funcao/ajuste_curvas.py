# Contém implementações de métodos numéricos para resolver sistemas lineares
from Metodos_Numericos import Sistemas_lineares

# Biblioteca para manipulação de matrizes e vetores
import numpy as np

# Biblioteca para manipulação de expressões matemáticas
import sympy as sp

def minimo_quadrado_discreto(pontos: list[float], fi: list[str]) -> sp.Expr:
    '''
    Função que implementa o método dos mínimos quadrados para ajuste de curvas a partir de pontos discretos.

    Argumentos:
    pontos (list): Lista representando os pontos no plano cartesiano. Cada ponto tem a forma [x, y],
                    onde x é a coordenada no eixo das abscissas e y é a coordenada no eixo das ordenadas.
                    exemplo: pontos = [[x1, y1], [x2, y2], ..., [xn, yn]]
    fi (list): Lista de strings representando as funções fi que serão utilizadas para ajustar os pontos.
               Cada string deve ser uma expressão válida em Python, com 'x' como a variável independente.
               exemplo: fi = ["1", "x", "x**2"]

    Retorno:
    F (sympy.core.expr.Expr): A expressão da funcão da curva de ajuste, representada como um objeto Sympy.
    '''
    n = len(fi)
    x = sp.symbols("x")
    fi = sp.sympify(fi)

    #Construçao da matriz A
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        for m in range(n):
            for i in range(len(pontos)):
                A[j][m] += fi[m].subs(x, pontos[i][0])*fi[j].subs(x, pontos[i][0])
    
    #Construçao do vetor b
    b = np.zeros(n, dtype=float)
    for j in range(n):
        for i in range(len(pontos)):
            b[j] += pontos[i][1]*fi[j].subs(x, pontos[i][0])
    
    #Resolvendo o sistema linear Ax = b
    if len(fi) > 1:
        meusistema = Sistemas_lineares.SistemaLinearSolver(A, b)
        solucao = meusistema.gradiente_conjugado(relatorio=True)
    else:
        solucao = [b[0]/A[0][0]]

    #Construindo a curva de ajuste a partir da soluçao do sistema acima
    F = 0
    for val, func_fi in zip(solucao, fi):
        F += val*func_fi

    return F

def minimo_quadrado_continuo(funcao: str, fi: list[str], a: float, b: float) -> sp.Expr:
    '''
    Função que implementa o método dos mínimos quadrados para ajuste de curvas a partir de uma funcao em um intervalo continuo [a, b].

    Argumentos:
    funcao (list): funcao representada por uma string em funcao de x no qual deseja ajustar.
                    exemplo: funcao = "x**2 + x - 1"
    fi (list): Lista de strings representando as funções fi que serão utilizadas para ajustar a funcao principal.
               Cada string deve ser uma expressão válida em Python, com 'x' como a variável independente.
               exemplo: fi = ["1", "x", "x**2"]
    a (float): intervalo inicial
    b (float): intervalo final

    Retorno:
    F (sympy.core.expr.Expr): A expressão da funcão da curva de ajuste, representada como um objeto Sympy.
    '''
    n = len(fi)
    x = sp.symbols("x")
    F = sp.sympify(funcao)
    fi = sp.sympify(fi)

    #Construçao da matriz A
    matriz_A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            matriz_A[i][j] += sp.integrate(fi[i]*fi[j],(x, a, b))
    
    #Construção do vetor b
    vetor_b = np.zeros(n, dtype=float) 
    for i in range(n):
        vetor_b[i] = sp.integrate(F*fi[i], (x, a, b))
    
    #Resolvendo o sistema linear Ax = b
    if len(fi) > 1:
        meusistema = Sistemas_lineares.SistemaLinearSolver(matriz_A, vetor_b)
        solucao = meusistema.gradiente_conjugado(relatorio=True)
    else:
        solucao = [vetor_b[0]/matriz_A[0][0]]
    
    F = 0
    for valor, func_fi in zip(solucao, fi):
        F += valor*func_fi

    return F

def interpolaçao(pontos: list[float]) -> sp.Expr:
    '''
    Função para realizar a interpolação polinomial de Lagrange.
    
    Argumentos:
    pontos (list): Pontos de interpolação no formato [[x1, y1], [x2, y2], ...].
                   Cada elemento da lista é um par ordenado representando um ponto (x, y).
    n (int): Número de pontos de interpolação.
    x (sympy.symbol): Variável simbólica representando o ponto de interpolação.
    l (sympy.Expr): Lista para armazenar os polinômios de Lagrange para cada ponto de interpolação.
                    É uma lista de objetos simbólicos do SymPy.
    P: Expressão simbólica representando o polinômio de Lagrange.

    retorno: Retorna o polinômio de Lagrange
    '''
    n = len(pontos)
    x = sp.symbols("x")
    l = np.ones(n, dtype=object)

    # Calculo dos polinômios de Lagrange para cada ponto de interpolação
    for i in range(n):
        for j in range(n):
            if i != j:
                l[i] *= (x - pontos[j][0])/(pontos[i][0] - pontos[j][0])

    P = 0
    for i in range(n):
        P += pontos[i][1]*l[i]
    
    return P.expand() # Retorna o polinômio de Lagrange expandido
