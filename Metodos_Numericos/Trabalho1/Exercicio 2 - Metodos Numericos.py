import pandas as pd
import numpy as np
from Metodos_Numericos import sistemas_lineares
from Metodos_Numericos import tempo_execuçao

def main():
    #importando a matriz e o vetor do arquivo em excel
    matriz_df = pd.read_excel("env/Metodos_Numericos/Trabalho1/Arquivos/Matriz_A.xlsx", header=None)
    vetor_df = pd.read_excel("env/Metodos_Numericos/Trabalho1/Arquivos/Vetor_b.xlsx", header=None)

    #transformando os dados no formato dataframe em uma array do tipo numpy
    matriz = matriz_df.to_numpy()
    vetor = vetor_df.to_numpy().reshape(-1)

    meusistema = sistemas_lineares.SistemaLinearSolver(matriz, vetor)

    meusistema.eliminaçao_gauss(relatorio=True)
    meusistema.jacobi(relatorio=True)
    meusistema.seidel(relatorio=True)
    meusistema.relaxamento(relatorio=True)
    #meusistema.gradiente_conjugado() 
    meusistema.gradiente_conjugado_quadrado(relatorio=True)

    #Condicionamento da matriz A36x36 para as seguintes normas:
    meusistema.norma(matriz, norma="l-norma", relatorio=True)
    meusistema.norma(matriz, norma="m-norma", relatorio=True)
    meusistema.norma(matriz, norma="k-norma", relatorio=True)

    #tempo medio das funçoes
    tempo_medio = { "Eliminaçao de gaus": tempo_execuçao(meusistema.eliminaçao_gauss),
                    "Jacobi":             tempo_execuçao(meusistema.jacobi),
                    "Seidel":             tempo_execuçao(meusistema.seidel),
                    "Relaxamento":        tempo_execuçao(meusistema.relaxamento),
                    "Conjugado quadrado": tempo_execuçao(meusistema.gradiente_conjugado_quadrado) }
    
    print(tempo_medio)

    return

if __name__ == "__main__":
    main()