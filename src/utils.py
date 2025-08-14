import numpy as np
from libs.ordering import get_index_matrix
from libs.ordering.sequency import sequency_order
from libs.row_wise import hadamard_row

def get_hadamard_patterns(img_size_h, img_size_w, n_measurements, n_hadamard=10, ordering="cake_cutting"):
    """
    Genera la matriz de codificación de apertura (CA) basada en patrones de Hadamard
    con el ordenamiento especificado.
    """
    M, N = img_size_h, img_size_w
    
    size = int(np.sqrt(2**n_hadamard))
    index_matrix = size*size - get_index_matrix(size, ordering)
    ordering_list = sequency_order(2**n_hadamard)

    order_temp = index_matrix.copy()
    order_temp[:, 1::2] = order_temp[::-1, 1::2]
    order_temp = index_matrix.reshape(-1, order="F")
    order_temp = np.argsort(order_temp)
    ordering_list = [ordering_list[i] for i in order_temp]

    H = []
    for i in range(2**n_hadamard):
        index = ordering_list[i]
        H.append(hadamard_row(index, n_hadamard))

    H = np.array(H).squeeze()
    initial_hadamard_ca = H[:n_measurements, :M*N]

    print(f"Generada CA de Hadamard con ordenamiento: {ordering}")
    return initial_hadamard_ca