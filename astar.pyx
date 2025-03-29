import heapq
from libc.math cimport abs

# Heurística de Manhattan
cdef inline int heuristic(int x1, int y1, int x2, int y2):
    return abs(x1 - x2) + abs(y1 - y2)

def astar(list start, list end, list grid):
    cdef:
        int rows = len(grid)
        int cols = len(grid[0])
        list open_list = []
        set closed_set = set()
        dict came_from = {}
        dict g_score = {}
        dict f_score = {}
        tuple current
        tuple neighbor
        int temp_g_score
        int temp_f_score
        heapq.heappush(open_list, (0, start))  # Heuristic cost + start position
        g_score[start] = 0
        f_score[start] = heuristic(start, end)

    while open_list:
        # Obtener el nodo con el menor coste f
        current = heapq.heappop(open_list)[1]

        if current == end:
            # Reconstruir el camino
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current)

        # Para cada vecino
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Verificar si el vecino está dentro de los límites
            if neighbor[0] < 0 or neighbor[0] >= rows or neighbor[1] < 0 or neighbor[1] >= cols:
                continue

            # Si el vecino es bloqueado o ya está en el conjunto cerrado, ignorarlo
            if grid[neighbor[0]][neighbor[1]] == 1 or neighbor in closed_set:
                continue

            temp_g_score = g_score.get(current, float('inf')) + 1
            temp_f_score = temp_g_score + heuristic(neighbor, end)

            # Si encontramos un mejor camino hacia el vecino
            if temp_f_score < f_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_f_score
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # Si no se encuentra camino