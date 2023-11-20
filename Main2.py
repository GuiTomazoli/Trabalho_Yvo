import numpy as np

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

    shadow_prices = table[0, 2:-3]
    optimal_profit = table[0, -1]
    optimal_point = table[1:3, -1]

    return optimal_point, optimal_profit, shadow_prices

# Novo exemplo de entrada
A = np.array([[3, 0], [0, 1], [-2, -5], [-4, -1], [1, 1]])
b = np.array([20, 45, -100, -45, 200])
c = np.array([-5, -9])

try:
    optimal_point, optimal_profit, shadow_prices = simplex_solver(A, b, c)
    print("Ponto ótimo:", optimal_point)
    print("Lucro ótimo:", optimal_profit)
    print("Preço-sombra de cada restrição:", shadow_prices)
except ValueError as e:
    print(e)
