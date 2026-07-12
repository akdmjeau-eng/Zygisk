import SPTAG
import numpy as np

# 1. Configurar los parámetros del índice
# Usamos 'BKT' (Balanced K-Means Tree) y vectores tipo Float (Float32)
algo = 'BKT'
data_type = 'Float'
dimension = 4  # Longitud de tus vectores
# 2. Inicializar el cliente de SPTAG
index = SPTAG.AnnIndex(algo, data_type, dimension)
# 3. Preparar datos de prueba (vectores aleatorios)
# SPTAG requiere que los vectores estén aplanados en una sola lista/arreglo continuo
vectors = np.array([
    [1.1, 2.2, 3.3, 4.4],
    [5.5, 6.6, 7.7, 8.8],
    [1.2, 2.3, 3.4, 4.5]
], dtype=np.float32)
# Convertir la matriz de NumPy a un formato plano compatible con la biblioteca
flat_vectors = vectors.flatten()
total_samples = len(vectors)
# 4. Construir el índice vectorial
# El último parámetro define si se calcula la distancia L2 (Euclidiana) o Coseno
index.Build(flat_vectors, total_samples, False) 
# 5. Realizar una búsqueda de Vecinos Más Cercanos (KNN)
query_vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
k_neighbors = 2  # Número de resultados que deseas obtener
# La función retorna una tupla: (IDs de los vectores, Distancias matemáticas)
results = index.Search(query_vector, k_neighbors)
print("IDs de los vecinos más cercanos:", results[0])
print("Distancias correspondientes:", results[1])
# 6. Guardar el índice en el disco para su uso posterior
index.Save("mi_indice_sptag")
