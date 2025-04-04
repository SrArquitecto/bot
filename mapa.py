import numpy as np
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
import cv2
import heapq
import pyautogui
import a_star
import time
import random
from queue import PriorityQueue
import time
from pathfinding.core.grid import Grid
from pathfinding.finder.bi_a_star import BiAStarFinder

class Mapa():
    def __init__(self):
        self.ruta_filtrada = []
        self.nodos_normalizados = []
        self.nodo_mas_cercano = None
        self.ruta = []
        self.matrix = np.zeros((350, 197), dtype=int)
        self.mapa_color = np.ones((350, 197), dtype=np.uint8) * 255
        self.mapa_navegacion = None
        self.finder = BiAStarFinder()
        self.grid = None

    def run(self, mask, coord_jugador, coord_nodos):
        # Generar el mapa
        self.matrix = np.zeros((350, 197), dtype=int)
        path = []
        self.ruta_filtrada = []
        coord_jugador = self.normalizar_coordenada(coord_jugador)
        if coord_nodos:
            coord_nodoN = self.normalizar_coordenadas(coord_nodos)
            self.generar_mapa(mask, coord_jugador, coord_nodos)
        else:
            # Si no hay coordenada de nodo, asegúrate de limpiar la ruta
            self.ruta_filtrada = []
            print("No hay nodos visibles, limpiando ruta y mapa.")
            self.generar_mapa(mask, coord_jugador)
            return self.ruta_filtrada
        # Generar el mapa
        closet = a_star.get_closest_target(coord_jugador, coord_nodoN, self.matrix)
        
        if closet:
            path = a_star.astar(coord_jugador, closet, self.matrix)  # Obtener el resultado del cálculo de A*

            if path:
                self.dibujar_ruta(path)
                self.ruta_filtrada = a_star.filter_nodes_by_distance(path, step=30)
                self._draw_nodes_on_map()
                self.ruta_filtrada = self.desnormalizar_coordenadas(self.ruta_filtrada)

                # Si la distancia es suficientemente pequeña, hacer una acción
                
            
        # Mostrar el mapa final
        self.mostrar_mapa()
        return self.ruta_filtrada

    def generar_mapa(self, mapa, posicion_jugador, nodos=()):
        if mapa.ndim == 2:
            self.mapa_color = cv2.cvtColor(mapa * 255, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("El mapa debe ser una imagen en escala de grises")
        
        if nodos:
             for nodo in nodos:
                cv2.circle(self.mapa_color, nodo, 5, (0, 0, 255), -1)
        
        cv2.circle(self.mapa_color, posicion_jugador, 5, (255, 0, 0), -1)

        # Mostrar el mapa generado para depuración
        self._mask_to_navigation_matrix()

    def mostrar_mapa(self):  
        cv2.imshow("Mapa Generado", self.mapa_color)

        
    # Función para leer la máscara RGB y convertirla a una matriz de navegación
    def _mask_to_navigation_matrix(self):
        # Convertir la imagen RGB a una matriz de valores
        self.matrix = np.zeros((self.mapa_color.shape[0], self.mapa_color.shape[1]), dtype=int)
        
        # Definir colores a mapear
        # (Verde) Área navegable
        self.matrix[np.all(self.mapa_color == [255, 255, 255], axis=-1)] = 0  # Área libre

        # (Rojo) Obstáculo
        self.matrix[np.all(self.mapa_color == [0, 0, 0], axis=-1)] = 1  # Obstáculo
        
        # (Azul) Objetivo
        self.matrix[np.all(self.mapa_color == [0, 0, 255], axis=-1)] = 2  # Objetivo

        # (Blanco) Inicio
        self.matrix[np.all(self.mapa_color == [255, 0, 0], axis=-1)] = 3  # Punto de inicio



    def euclidean_distance(self, start, end):
        return np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

    # Obtener el objetivo más cercano
    def dibujar_ruta(self, path):
        if path:  
            for i in range(len(path) - 1):
                start_point = (path[i][0], path[i][1])
                end_point = (path[i + 1][0], path[i + 1][1])
                cv2.line(self.mapa_color, start_point, end_point, (0, 255, 255), 2)
                

    def _extraer_ruta(self, ruta):
        self.ruta = []
        if ruta:
            for punto in ruta:
                self.ruta.append((punto[0], punto[1]))

    # -------------------------------
    # Dibujo de nodos en el mapa
    # -------------------------------
    def _draw_nodes_on_map(self, color=(0, 255, 0), radius=3):
        """Dibuja los nodos filtrados en el mapa."""
        for node in self.ruta_filtrada:
            cv2.circle(self.mapa_color, node, radius, color, -1)
        return self.mapa_navegacion

    # -------------------------------
    # Centrado directo de la cámara en X
    # -------------------------------
    def _align_camera_to_node(self, node_center_x, tolerance=5):
        """Mueve el ratón de forma gradual hacia el centro del nodo, comprobando si está centrado."""
        # Obtener las dimensiones de la pantalla
        center_x, center_y = 1920 // 2, 1080 // 2


            # Calcular la diferencia entre el centro de la pantalla y el centro del objeto
        delta_x = center_x - node_center_x
        print(delta_x)
            # Si el objeto ya está suficientemente centrado, no mover el ratón
        if abs(delta_x) < tolerance:
            print("El objeto ya está centrado.")
            return

            # Determinar la dirección en la que mover el ratón
        if delta_x > 0:
            # Si el objeto está a la izquierda del centro de la pantalla, mover a la derecha
            pyautogui.move(-delta_x, 0, 0.5)
            print(f"Moviendo ratón 50 píxeles a la derecha.")
        else:
                # Si el objeto está a la derecha del centro de la pantalla, mover a la izquierda
            pyautogui.move(-delta_x, 0, 0.5)
            print(f"Moviendo ratón 50 píxeles a la izquierda.")
                # -------------------------------
                # Comprobación de nodo fuera de la vista

    def _align_camera_x_to_node2(self, node_x):
        """Mueve el ratón directamente para centrar el nodo en una sola iteración."""
        screen_width, _ = pyautogui.size()
        center_x = screen_width // 2

        # Movimiento directo en lugar de proporcional
        pyautogui.moveTo(node_x, pyautogui.position().y, 0)
    # -------------------------------
    # Comprobación de nodo fuera de la vista
    # -------------------------------
    def _is_node_in_view(self, node_x, margin=0.1):
        """Verifica si el nodo está dentro de una zona segura en pantalla."""
        screen_width, _ = pyautogui.size()
        left_limit = screen_width * margin
        right_limit = screen_width * (1 - margin)

        return left_limit < node_x < right_limit


    def normalizar_coordenadas(self, nodos, w=350, h=197, orig_w=1920, orig_h=1080):
        """
        Normaliza las coordenadas de una resolución original (orig_w, orig_h) a una nueva resolución (w, h).
        """
        nodos_normalizados = []
        for nodo in nodos:
            x, y = nodo  # Desempaquetar el nodo
            x = x / orig_w * w    # Normalizar a la nueva resolución
            y = y / orig_h * h
            nodos_normalizados.append((int(x), int(y)))  # Añadir las coordenadas normalizadas
        return nodos_normalizados

    def desnormalizar_coordenadas(self, nodos, w=350, h=197, orig_w=1920, orig_h=1080):
        """
        Desnormaliza las coordenadas de una resolución de (w, h) a una resolución original (orig_w, orig_h).
        """
        nodos_desnormalizados = []
        for nodo in nodos:
            x, y = nodo  # Desempaquetar el nodo
            x = x / w * orig_w    # Desnormalizar a la resolución original
            y = y / h * orig_h
            nodos_desnormalizados.append((int(x), int(y)))  # Añadir las coordenadas desnormalizadas
        return nodos_desnormalizados

    def normalizar_coordenada(self, nodo, w=350, h=197, orig_w=1920, orig_h=1080):
        """
        Normaliza una única coordenada de una resolución original (orig_w, orig_h) a una nueva resolución (w, h).
        """
        x, y = nodo  # Desempaquetar el nodo
        x = x / orig_w * w    # Normalizar a la nueva resolución
        y = y / orig_h * h
        return (int(x), int(y))  # Devolver las coordenadas normalizadas

    def desnormalizar_coordenada(self, nodo, w=350, h=197, orig_w=1920, orig_h=1080):
        """
        Desnormaliza una única coordenada de una resolución de (w, h) a una resolución original (orig_w, orig_h).
        """
        x, y = nodo  # Desempaquetar el nodo
        x = x / w * orig_w    # Desnormalizar a la resolución original
        y = y / h * orig_h
        return (int(x), int(y))  # Devolver las coordenadas desnormalizadas

