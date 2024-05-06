#Imports nativos do python
import __main__
import os

#Biblioteca de manipulação de texto
import textwrap

#Bibliotecas de analise de dados e manipulaçao de matrizes
import numpy as np
import pandas as pd


class SistemaLinearSolver:

    def __init__(self, A: np.ndarray, b: np.ndarray, e: float = 1e-4):
        self.A = A
        self.b = b
        self.e = e
        self.linha = A.shape[0]
        self.coluna = A.shape[1]
        self.x = None

    def salvar_em_excel(self, A: np.ndarray, nome_planilha: str) -> None:

        nome_arquivo = os.path.basename(__main__.__file__).replace("py", "xlsx")
        # Criar um DataFrame do Pandas com a matriz
        df = pd.DataFrame(A)

        # Verifica se existe o arquivo em excel, caso nao exita cria um
        if not os.path.isfile(nome_arquivo):
            with pd.ExcelWriter(nome_arquivo) as writer:
                df.to_excel(writer, sheet_name=nome_planilha)

        # Cria a guia com os dados passado por parametro
        with pd.ExcelWriter(nome_arquivo, mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=nome_planilha, index=False, header=False)

        return
    
    def relatorio(self, nome_funcao: str, x: np.ndarray[float], n_interaçoes: int=None, erro_relativo: float=None, ) -> None:

        print(textwrap.dedent(f"""
                Metodo {nome_funcao} para um sistema linear Ax = b:
                Erro absoluto ||x(k) - x(k-1)|| = {erro_relativo}
                Número de interações = {n_interaçoes}
                Vetor solução aproximado (x) ="""))
        print(x)

    def norma(self, A: np.ndarray, *, norma: str="l-norma", relatorio: bool=False) -> float:

        # Calcula diferentes normas para uma matriz:
        norma_value = 0

        match norma:

            case "l-norma":
                if np.ndim(A) == 2: # Para uma matriz (dimensao = 2)
                    for i in A:
                        x = 0
                        for j in i:
                            x += abs(j)
                        if x > norma_value:
                            norma_value = x
                else:
                    for i in A: # Para um vetor (dimensao = 1)
                        norma_value += abs(i)


            case "m-norma":
                B = A.transpose()
                for i in B:
                    x = 0
                    for j in i:
                        x += abs(j)
                    if x > norma_value:
                        norma_value = x

            case "k-norma":
                if np.ndim(A) == 2: # Para uma Maitrz
                    for i in A:
                        for j in i:
                            norma_value += j**2
                else: # Para um vetor
                    for i in A:
                        norma_value += i**2
                norma_value = norma_value**(1/2)

            case _:
                print("Norma inválida")
                return

        if relatorio:
            print(f"{norma} = {norma_value}")

        return norma_value

    def matriz_alfa(self) -> tuple[np.ndarray, np.ndarray]:
        "Transforma o sistema linear Ax = b em um sistema interativo do tipo x = Cx + g"

        linha = self.A.shape[0] #numero de linhas da matriz
        coluna = self.A.shape[1]

        C = np.zeros((linha, coluna))
        g = self.b.copy()

        for i in range(linha):
            for j in range(coluna):
                if i != j:
                    C[i][j] = -self.A[i][j]/self.A[i][i]
            g[i] = self.b[i]/self.A[i][i]

        return C, g

    def eliminaçao_gauss(self, *, relatorio: bool=False, salvar: bool=False) -> np.ndarray[float]:
        '''
        Implementação do método de eliminaçao de Gauss para um sistema linear Ax = b.
        A: Matriz de coeficientes do sistema;
        b: Vetor de termos independentes do sistema;
        A_b: Matriz aumentada
        x: vetor solução do sistema. '''

        A_b = np.hstack((self.A, self.b[:, np.newaxis]))
        linha = self.linha

        for i in range(linha-1):
            pivo = A_b[i][i]
            for j in range(i+1, linha):
                A_b[j] = A_b[j] - (A_b[j][i]/pivo)*A_b[i]

        tolerancia = 1e-12
        A_b[np.abs(A_b) < tolerancia] = 0

        x = np.zeros(self.linha)

        for i in range(linha - 1, -1, -1):
            soma = 0
            pivo = A_b[i][i]
            if pivo == 0:
                print(f'pivo igual a zero na linha {i}')
                return
            for j in range(linha):
                soma += A_b[i][j]*x[j]
            x[i] = (A_b[i][linha] - soma)/pivo

        #Exibir informaçoes na tela do terminal
        if relatorio:
            nome_funcao = "Eliminação de Gauss"
            self.relatorio(nome_funcao, x)

        #Salvar o resultado em um planilha
        if salvar:
            self.salvar_em_excel(A_b, "Matriz aumentada")
            self.salvar_em_excel(x, "Eliminaçao de gaus")

        self.x = x
        return x

    def jacobi(self, *, relatorio: bool=False, salvar: bool=False) -> np.ndarray[float]:
        '''
        Implementação do método de Jacobi para resolver um sistema linear Ax = b.
        reescrevendo o sistema da seguinte maneira: x = Cx + g.
        A: Matriz de coeficientes do sistema;
        b: Vetor de termos independentes do sistema;
        e: Erro para o critério de parada (padrão = 0.001);
        x: vetor solução do sistema aproximado. '''

        linha = self.linha #numero de linhas da matriz
        coluna = self.coluna #numero de colunas da matriz

        x = np.zeros(linha)
        xk = np.zeros(linha)

        #Criando a matriz C e vetor g:
        C, g = self.matriz_alfa()

        #Determinando o vetor soluçao (x) do sistema por interaçoes:
        erro = 1 # erro para a condiçao de parada
        n_interaçoes = 0
        while(erro >= self.e):
            xk = np.zeros(linha)
            for i in range(linha):
                for j in range(coluna):
                    xk[i] += C[i][j] * x[j]
                xk[i] += g[i]

            erro = self.norma(xk - x, norma="l-norma")
            x = xk.copy()
            n_interaçoes += 1

        if relatorio:
            self.relatorio(x, "Jacobi")

        if salvar:
            nome_funcao = "Jacobi"
            self.relatorio(nome_funcao, x, n_interaçoes, erro)

        return x

    def seidel(self, *, relatorio: bool=False, salvar: bool=False) -> np.ndarray[float]:
        '''
        Implementação do método de Gauss-Seidel para resolver um sistema linear matricial Ax = b.
        reescrevendo o sistema da seguinte maneira: x = Cx + g.
        A: Matriz de coeficientes do sistema;
        b: Vetor de termos independentes do sistema;
        e: Erro para o critério de parada (padrão = 0.001);
        x: vetor solução do sistema aproximado. '''

        linha = self.linha #numero de linhas da matriz
        coluna = self.coluna #numero de colunas da matriz

        x = np.zeros(linha)
        xk = np.zeros(linha)

        #Criando a matriz C e vetor g:
        C, g = self.matriz_alfa()

        #Determinando o vetor soluçao do sistema x por interaçoes:
        erro_relativo = 1 # erro para a condiçao de parada
        n_interaçoes = 0
        while(erro_relativo >= self.e):
            xk = np.zeros(linha)
            xt = x.copy()
            for i in range(linha):
                for j in range(coluna):
                    xk[i] += C[i][j] * x[j]
                xk[i] += g[i]
                x[i] = xk[i]

            x = xk.copy()
            erro_relativo = self.norma(x - xt, norma="l-norma")
            n_interaçoes += 1

        if relatorio:
            nome_funcao = "Gauss-Seidel"
            self.relatorio(nome_funcao, x, n_interaçoes, erro_relativo)

        if salvar:
            self.salvar_em_excel(x, "Seidel")

        return x

    def relaxamento(self, *, relatorio: bool=False, salvar: bool=False) -> np.ndarray[float]:

        linha = self.linha
        coluna = self.coluna

        C = np.zeros((linha, coluna))
        d = self.b.copy()

        for i in range(linha):
            d[i] = d[i]/self.A[i][i]
            for j in range(coluna):
                C[i][j] = -self.A[i][j]/self.A[i][i]
        Ct = np.transpose(C) #Transporta de C

        x = np.zeros(linha)
        xk = np.zeros(linha)
        R = d - x + (C @ x) # residuo

        erro_absoluto = 1
        n_interaçoes = 0

        while(erro_absoluto >= self.e):
            n_max = np.argmax(np.abs(R))
            delta_x = R[n_max]

            R = R + delta_x*Ct[n_max]
            xk[n_max] += delta_x
            erro_absoluto = self.norma(xk - x, norma="l-norma")

            x = np.copy(xk)
            n_interaçoes += 1

        if relatorio:
            nome_funcao = "Relaxamento"
            self.relatorio(nome_funcao, x, n_interaçoes, erro_absoluto)

        if salvar:
            self.salvar_em_excel(x, "Relaxamento")

        return x

    def gradiente_conjugado(self, *, relatorio: bool=False, salvar: bool=False) -> np.ndarray[float]:

        x = np.zeros(self.linha)
        r = self.b - self.A @ x
        p = np.copy(r)

        erro_absoluto = 1
        n_interaçoes = 0

        while(erro_absoluto >= self.e):

            alfa = (r @ r)/(p @ (self.A @ p))
            xk = x + alfa*p
            rk = r - alfa*(self.A @ p)
            beta = (rk @ rk)/(r @ r)
            pk = rk + beta*p

            erro_absoluto = self.norma(xk - x, norma="l-norma")
            n_interaçoes += 1

            #Atualizar valores
            x = np.copy(xk)
            r = np.copy(rk)
            p = np.copy(pk)

        if relatorio:
            nome_funcao = "Gradiente conjugado"
            self.relatorio(nome_funcao, x, n_interaçoes, erro_absoluto)

        if salvar:
            self.salvar_em_excel(x, "Gradiente Conjugado")

        return x

    def gradiente_conjugado_quadrado(self, *, relatorio: bool=False, salvar: bool=False) -> np.ndarray[float]:

        linha = self.linha

        x = np.zeros(linha)
        xk = np.zeros(linha)
        r0 = self.b - self.A @ x
        rk = np.copy(r0)
        u = np.zeros(linha)
        w = np.zeros(linha)
        alfa = 1
        sigma = 1

        erro = 1
        n_interaçoes = 0

        while(erro >= self.e):
            p = rk @ r0
            beta = -1/alfa*(p/sigma)
            v = rk - beta*u
            w = v - beta*(u-beta*w)
            c = self.A @ w
            sigma = c @ r0
            alfa = p/sigma
            u = v - alfa*c
            xk = x + alfa*(v + u)
            rk = rk - alfa*(self.A @ (v + u))

            erro = self.norma(xk - x)
            n_interaçoes += 1
            x = np.copy(xk)

        if relatorio:
            nome_funcao = "Gradiente Conjugado Quadrado"
            self.relatorio(nome_funcao, x, n_interaçoes, erro)

        if salvar:
            self.salvar_em_excel(x, "Gradiente Conjugado Quadrado")

        self.x = x
        return x
