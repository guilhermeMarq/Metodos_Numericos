from Metodos_Numericos import sistemas_nao_lineares
import numpy as np

def main():
    funcao = ["(x0-1)**2 + (x1-1)**2 + (x2-1)**2 - 1",
              "2*x0**2 + (x1-1)**2 -4*x2",
              "3*x0**2 + 2*x2**2 - 4*x1"]
    
    #x0 = np.array([0, 0, 0])
    x0 = np.array([2, 2, 2])
    
    sistemas_nao_lineares.metodo_newton(funcao, x0)

if __name__ == "__main__":
    main()