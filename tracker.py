import cv2
import numpy as np

class SeguimientoNodos:
    def __init__(self):
        self.nodo_anterior = None  # Última posición conocida del nodo
        self.pos_relativa = None   # Posición relativa al jugador

    def _convertir_a_relativo(self, nodo, jugador):
        """Convierte la posición absoluta del nodo a coordenadas relativas al jugador."""
        return (nodo[0] - jugador[0], nodo[1] - jugador[1])

    def _convertir_a_absoluto(self, nodo_relativo, jugador):
        """Convierte una posición relativa a absoluta tras mover la cámara."""
        return (nodo_relativo[0] + jugador[0], nodo_relativo[1] + jugador[1])

    def encontrar_nodo_mas_cercano(self, nodos_detectados, jugador):
        """Encuentra el nodo más cercano a la última posición conocida."""
        if not self.nodo_anterior:
            return min(nodos_detectados, key=lambda nodo: np.linalg.norm(np.array(nodo) - np.array(jugador)))

        # Convertimos la posición relativa a la nueva posición absoluta tras el movimiento de la cámara
        nodo_esperado = self._convertir_a_absoluto(self.pos_relativa, jugador)

        return min(nodos_detectados, key=lambda nodo: np.linalg.norm(np.array(nodo) - np.array(nodo_esperado)))

    def actualizar_nodo(self, nodos_detectados, jugador):
        """
        Actualiza el nodo objetivo tras la rotación de la cámara.
        """
        nodo_actualizado = self.encontrar_nodo_mas_cercano(nodos_detectados, jugador)

        if nodo_actualizado:
            self.nodo_anterior = nodo_actualizado
            self.pos_relativa = self._convertir_a_relativo(nodo_actualizado, jugador)

        return nodo_actualizado
    

    def dibujar(self, img, nodo):
        if nodo is not None:
            cv2.circle(img, nodo, 5, (0, 0, 255), -1)
        cv2.imshow("Track", img)

if __name__ == "__main__":
    jugador_pos = (960, 540)  # Centro de la pantalla

    # Detección inicial de nodos
    nodos_detectados_1 = [(500, 400), (800, 600)]  # Lista de coordenadas de nodos cian

    # Creamos el sistema de seguimiento de nodos
    tracker = SeguimientoNodos()

    # Primera detección
    nodo_seleccionado = tracker.actualizar_nodo(nodos_detectados_1, jugador_pos)
    print(f"Nodo inicial seleccionado: {nodo_seleccionado}")

    # Simulamos movimiento de la cámara (cambia la posición de los nodos en la imagen)
    jugador_pos = (950, 540)  # La cámara se mueve un poco hacia la izquierda
    nodos_detectados_2 = [(490, 400), (790, 600)]  # Nueva detección de nodos

    # Seguimiento después de la rotación de la cámara
    nodo_seleccionado = tracker.actualizar_nodo(nodos_detectados_2, jugador_pos)
    print(f"Nodo tras la rotación: {nodo_seleccionado}")