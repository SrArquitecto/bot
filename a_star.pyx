import heapq
import numpy as np
cimport numpy as np
from libc.math cimport INFINITY
from cython.parallel import prange

# Función que verifica si un punto está bloqueado en la matriz
cdef bint is_blocked(tuple point, np.ndarray matrix):
    cdef int x = point[0]
    cdef int y = point[1]
    return matrix[y, x] == 1  # 1 representa un obstáculo en el mapa

# Obtener el objetivo más cercano

cpdef object get_closest_target(tuple start, list targets, np.ndarray matrix):
    if not targets:
        return None  # Evitar iteración innecesaria si targets está vacío

    cdef float min_distance = INFINITY
    cdef object closest_target = None  # Usamos 'object' en lugar de 'tuple'
    cdef float dist
    cdef int i

    # Se usa range en lugar de prange para evitar conflictos con el GIL
    for i in range(len(targets)):  
        target = targets[i]
        if is_blocked(target, matrix):  
            continue
        dist = euclidean_distance(start, target) + 0.5 * abs(start[0] - target[0])
        if dist < min_distance:
            min_distance = dist
            closest_target = target  

    return closest_target

# Función para obtener los vecinos del nodo actual (optimizada en Cython)
cdef list get_neighbors(tuple node, np.ndarray matrix):
    cdef int x, y, nx, ny
    cdef list neighbors = []
    cdef list directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (1, 1), (1, -1), (-1, 1)]
    
    x, y = node

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < matrix.shape[1] and 0 <= ny < matrix.shape[0]:
            neighbors.append((nx, ny))
    
    return neighbors

# Implementación del algoritmo A* (optimizada en Cython)
cpdef list astar(tuple start, tuple goal, np.ndarray matrix):
    cdef list open_list = []
    cdef dict[tuple, int] g_costs = {start: 0}  # Costo acumulado desde el inicio
    cdef dict[tuple, float] f_costs = {start: euclidean_distance(start, goal)}  # Costo total estimado
    cdef set[tuple] visited = set()  # Conjunto de nodos visitados
    cdef dict[tuple, tuple] came_from = {}  # Conjunto de nodos visitados
    cdef tuple current
    cdef list neighbors
    cdef tuple neighbor
    cdef int tentative_g_cost
    cdef list path = []

    heapq.heappush(open_list, (f_costs[start], start))  # Inicializamos la cola de prioridad con el nodo de inicio

    while open_list:
        _, current = heapq.heappop(open_list)  # Obtenemos el nodo con el costo más bajo

        if current == goal:
            # Reconstruir la ruta desde el nodo final
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Devuelve la ruta invertida (de start a goal)

        visited.add(current)

        # Obtenemos los vecinos del nodo actual
        neighbors = get_neighbors(current, matrix)

        for neighbor in neighbors:
            if neighbor in visited or is_blocked(neighbor, matrix):  # Ignorar los nodos ya visitados o bloqueados
                continue

            tentative_g_cost = g_costs[current] + 1  # Costo acumulado al vecino

            # Si es un nodo nuevo o encontramos una mejor ruta
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                came_from[neighbor] = current
                g_costs[neighbor] = tentative_g_cost
                f_costs[neighbor] = tentative_g_cost + euclidean_distance(neighbor, goal)
                heapq.heappush(open_list, (f_costs[neighbor], neighbor))  # Agregar el vecino a la cola

    return None  # No se encontró ruta

# Función para filtrar los nodos por distancia (optimizada en Cython)
cpdef list filter_nodes_by_distance(list path, int step=100):
    cdef list ruta_filtrada = []
    cdef tuple last_node, node
    cdef float dist

    if path:
        ruta_filtrada = [path[0]]
        last_node = path[0]

        for node in path[1:]:
            dist = euclidean_distance(last_node, node)  # Calculamos la distancia Euclidiana
            if dist >= step:
                ruta_filtrada.append(node)
                last_node = node

        if ruta_filtrada[-1] != path[-1]:
            ruta_filtrada.append(path[-1])

    return ruta_filtrada

# Función para calcular la distancia Euclidiana entre dos puntos (optimizada en Cython)
cpdef float euclidean_distance(tuple start, tuple end):
    cdef float dx = start[0] - end[0]
    cdef float dy = start[1] - end[1]
    return (dx * dx + dy * dy) ** 0.5

# Optimización adicional usando caching de distancias
# Caching de la distancia entre los nodos cercanos
cpdef dict calculate_distances(tuple start, list nodes):
    cdef dict distances = {}
    cdef tuple node
    for node in nodes:
        distances[node] = euclidean_distance(start, node)
    return distances
