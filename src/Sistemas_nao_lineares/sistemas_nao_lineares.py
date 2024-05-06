import numpy as np
import sympy as sp
import textwrap

def metodo_newton(x0: np.ndarray, e: float = 0.001) -> np.ndarray:

    '''
    Implementação do método de Newton para resolver sistemas não lineares.
    Reescrevendo o sistema da seguinte maneira: J.s = -F e x = x + s
    F: Função nao linear;
    x0: Chute inicial;
    J: Matriz jacobiana;
    s: Vetor de correção;
    e: Erro para o critério de parada (padrão = 0.001). '''

    linha = len(x0)
    variaveis = f"x0:{linha}"

    x = sp.symbols(variaveis, real=True)
    F = np.empty(linha, dtype=object)
    F[0] = x[0]**3 + 2*x[1] + x[2] - 4
    F[1] = 2*x[0]**2 + x[1]**2 - 4*x[2] + 1
    F[2] = 3*x[0]**2 - 4*x[1] + x[2]

    #Criando a Matriz Jacobiana (J):
    J = np.empty((linha,linha), dtype=object)

    for m in range(linha):
        for n in range(linha):
            J[m][n] = sp.diff(F[m], x[n])

    #Resolvendo a Matriz jacobiana com o chute inicial:
    erro = 1 
    n_interaçoes = 0
    while(e <= erro):
        n_interaçoes += 1
        Jk = np.copy(J)
        for m in range(linha):
            for n in range(linha):
                Jk[m][n] = J[m][n].subs(x[n], x0[n])

        #Criando um dicionario para colocar no sympy.subs(dic)
        dicionario = {}   
        for var, valor in zip(x, x0):
            dicionario[var] = valor # {x0: valor0, x1: valor1, ...}

        #Calculando Fk
        Fk = np.empty(linha)
        for i, var in enumerate(F):
            Fk[i] = var.subs(dicionario)

        #Calculando s
        Jk = Jk.astype(float)
        s = -np.linalg.inv(Jk) @ Fk #s = -J^-1*F
        x0 = x0 + s

        #Calculo do Erro:
        erro = np.linalg.norm(s)

    print(textwrap.dedent(f"""
          Metodo de Newton para sistemas nao lineares:
          Vetor solução aproximado (x) = {x0}
          Erro ||s(k)|| = {erro}
          Número de interações = {n_interaçoes}
          Matriz Jacobiana:"""))
    sp.pretty_print(sp.Matrix(J))
    
    return x0