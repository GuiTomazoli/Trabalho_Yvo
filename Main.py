import numpy as np

# Configuração para imprimir números em float
np.set_printoptions(precision=2, suppress=True)

def simplex_solver(A, b, c):
    # Adiciona variáveis de folga
    A = np.hstack((A, np.eye(len(b))))
    c = np.concatenate((c, np.zeros(len(b))))

    # Inicializa a tabela Simplex
    table = np.vstack((np.hstack((c, [0])), np.hstack((A, np.reshape(b, (len(b), 1))))))

    while np.min(table[0, :-1]) < 0:
        # Encontra a coluna pivô
        pivot_col = np.argmin(table[0, :-1])

        # Verifica se a coluna pivô permite uma escolha adequada da linha pivô
        if np.all(table[1:, pivot_col] <= 0):
            raise ValueError("O problema é ilimitado: a coluna pivô não permite uma escolha adequada da linha pivô.")

        # Encontra a linha pivô
        ratios = table[1:, -1] / table[1:, pivot_col]
        ratios[table[1:, pivot_col] <= 0] = np.inf  # Evita a divisão por zero
        pivot_row = np.argmin(ratios) + 1

        # Atualiza a tabela Simplex
        table[pivot_row, :] /= table[pivot_row, pivot_col]
        for i in range(len(table)):
            if i != pivot_row:
                table[i, :] -= table[i, pivot_col] * table[pivot_row, :]

    
    # Extrai as soluções do exemplo 1
    quadro_otimo = table
    lucro_otimo = table[0, -1]
    valores_otimos = table[2:, -1]
    preco_sombra = table[0, 2:-1]
    '''
    # Extrai as soluções do exemplo 2
    quadro_otimo = table
    lucro_otimo = table[0, -1]
    valores_otimos = table[1:3, -1]
    preco_sombra = table[0, 2:-1]
    '''
    return quadro_otimo, lucro_otimo, valores_otimos, preco_sombra

# Exemplo de entrada 1
A = np.array([[3, 0], [0, 1.5], [0.25, 0.5]]) #Restrições
b = np.array([250, 100, 50]) #Lado Direito
c = np.array([-5, -7]) #MaximizeZ
'''
# Exemplo de entrada 2
A = np.array([[3, 0], [0, 1], [-2, -5], [-4, -1], [1, 1]]) #Restrições
b = np.array([20, 45, -100, -45, 200]) #Lado Direito
c = np.array([-5, -9]) #MaximizeZ
'''
try:
    quadro_otimo, lucro_otimo, valores_otimos, preco_sombra = simplex_solver(A, b, c)
    print("Quadro ótimo:", "\n", quadro_otimo)
    print("Valores ótimos:", valores_otimos)
    print("Lucro ótimo:", lucro_otimo)
    print("Preço-sombra de cada restrição:", preco_sombra)
except ValueError as e:
    print(e)
