# astar.pyx

import heapq
import numpy as np
cimport numpy as np

# Función que verifica si un punto está bloqueado en la matriz
cdef bint is_blocked(tuple point, np.ndarray matrix):
    cdef int x = point[0]
    cdef int y = point[1]
    return matrix[y, x] == 1  # 1 representa un obstáculo en el mapa

# Obtener el objetivo más cercano
cpdef tuple get_closest_target(tuple start, list targets, np.ndarray matrix):
    cdef tuple closest_target = None
    cdef float min_distance = float('inf')
    cdef float dist
    # Recorrer los objetivos en la lista de Python
    for target in targets:
        if is_blocked(target, matrix):  # Si el objetivo está bloqueado, lo ignoramos
            continue
        
        dist = euclidean_distance(start, target)
        if dist < min_distance:
            min_distance = dist
            closest_target = target
    
    return closest_target  # Devuelve None si no hay objetivos válidos

# Función para obtener los vecinos del nodo actual (optimizada en Cython)
cdef list get_neighbors(tuple node, np.ndarray matrix):
    x, y = node
    cdef list neighbors = []
    cdef int nx, ny
    cdef list directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (1, 1), (1, -1), (-1, 1)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < matrix.shape[1] and 0 <= ny < matrix.shape[0]:
            neighbors.append((nx, ny))
    
    return neighbors

# Implementación del algoritmo A* (optimizada en Cython)
cpdef list astar(tuple start, tuple goal, np.ndarray matrix):
    cdef list open_list = []
    heapq.heappush(open_list, (0, start))  # Colocamos el nodo inicial
    cdef dict came_from = {}
    cdef dict g_costs = {start: 0}  # Costo acumulado desde el inicio
    cdef dict f_costs = {start: euclidean_distance(start, goal)}  # Costo total estimado
    cdef set visited = set()  # Conjunto de nodos visitados
    cdef tuple current
    cdef list neighbors
    cdef tuple neighbor
    cdef int tentative_g_cost
    cdef list path = []

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
            if neighbor in visited:  # Ignorar los nodos ya visitados
                continue

            if is_blocked(neighbor, matrix):  # Si el vecino está bloqueado, lo ignoramos
                continue

            tentative_g_cost = g_costs[current] + 1  # Costo acumulado al vecino

            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                came_from[neighbor] = current
                g_costs[neighbor] = tentative_g_cost
                f_costs[neighbor] = tentative_g_cost + euclidean_distance(neighbor, goal)
                heapq.heappush(open_list, (f_costs[neighbor], neighbor))  # Agregar el vecino a la cola

    return None  # No se encontró ruta


cpdef list filter_nodes_by_distance(list path, int step=100):
    cdef list ruta_filtrada = []
    cdef tuple last_node, node
    cdef float dist
     # Declaramos 'dist' aquí, antes del ciclo

    if path:
        # Inicia con el primer nodo (posición inicial)
        ruta_filtrada = [path[0]]
        last_node = path[0]

        # Filtrar los nodos basados en la distancia mínima (step)
        for node in path[1:]:
            dist = euclidean_distance(last_node, node)  # Calculamos la distancia Euclidiana
            if dist >= step:
                ruta_filtrada.append(node)
                last_node = node

        # Aseguramos que el último nodo de la ruta esté incluido si no fue agregado
        if ruta_filtrada[-1] != path[-1]:
            ruta_filtrada.append(path[-1])
    
    return ruta_filtrada

# Función para calcular la distancia Euclidiana entre dos puntos (optimizada en Cython)
cdef float euclidean_distance(tuple start, tuple end):
    cdef float dx = start[0] - end[0]
    cdef float dy = start[1] - end[1]
    return (dx * dx + dy * dy) ** 0.5

# Función para filtrar los nodos por distancia (optimizada en Cython)
