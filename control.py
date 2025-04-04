import time
import threading
import os
import mss
import numpy as np
import cv2
from pynput import keyboard
from datetime import datetime
from deteccion import ColorModel, ColorModelInterface
from segmentacion import MaskGeneratorInterface, BinaryMaskGenerator
from mapa import Mapa
from concurrent.futures import ThreadPoolExecutor
from evdev import UInput, ecodes
import vgamepad as vg
import pyautogui
import torch
torch.cuda.empty_cache()
import a_star
from tracker import SeguimientoNodos

class TeclaControl:
    def __init__(self):
        self.ui = UInput({
            ecodes.EV_KEY: [ecodes.KEY_W, ecodes.BTN_LEFT, ecodes.KEY_E, ecodes.KEY_3, ecodes.KEY_SPACE, ecodes.KEY_D],  # Tecla 'W' y botón izquierdo del ratón
            ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y],  # Movimientos del ratón
        }, name="VirtualMouseKeyboard")
        self.moverse = False  # Variable de control para saber si debemos movernos o no
        self.luchando = False
        self.tecla = 'w'  # Tecla que queremos presionar
        self.finw = False
        self.finl = False
        self.coger = False
        self.hilo = threading.Thread(target=self.controlar_tecla)  # Hilo para controlar la tecla
        self.hilo.daemon = True  # El hilo se cierra automáticamente cuando el programa termina
        self.hilo.start()
        self.coord = None

        # Inicializar el dispositivo de entrada para teclado y ratón

    # Para agregar pausas en el bucle
    def toque(self):
        self.ui.write(ecodes.EV_KEY, ecodes.KEY_D, 1)
        self.ui.syn() 
        self.ui.write(ecodes.EV_KEY, ecodes.KEY_D, 0)
        self.ui.syn() 

    def ejecutar_secuencia_bloqueo(self):
        self.ui.write(ecodes.EV_KEY, ecodes.KEY_SPACE, 1)
        self.ui.syn() 
        self.ui.write(ecodes.EV_KEY, ecodes.KEY_SPACE, 0)
        self.ui.syn() 


    def align_camera_to_node(self, node_center_x):
        """Mueve el ratón de forma gradual hasta centrar el nodo con un bucle for."""
        screen_center_x = 1920 // 2  # Centro de la pantalla en X
        # Calcular la diferencia entre el nodo y el centro de la pantalla
        delta_x = node_center_x - screen_center_x  # (Positivo = Derecha, Negativo = Izquierda)


        # Determinar el número máximo de pasos (por ejemplo, 100 pasos)
        max_steps = 100
        max_step = 2  # Tamaño máximo del paso por iteración

        # Bucle for para mover el ratón gradualmente
        for i in range(max_steps):
            # Si la diferencia es pequeña, se puede salir antes
            if abs(delta_x) < 1:
                print("[INFO] El nodo ya está centrado. Terminando alineación.")
                break

            # Calcular el paso en cada iteración
            step = min(abs(delta_x), max_step)
            move_x = step if delta_x > 0 else -step  # Determina la dirección del movimiento

            # Enviar el evento de movimiento del ratón
            try:
                self.ui.write(ecodes.EV_REL, ecodes.REL_X, move_x)
                self.ui.syn()
                print("[SUCCESS] Movimiento enviado correctamente.")
            except Exception as e:
                print(f"[ERROR] Ocurrió un error al mover el ratón: {e}")
                break  # Salir del bucle si ocurre un error

            # Actualizar la posición del nodo (simulando el movimiento)
            node_center_x += move_x

            # Calcular la nueva diferencia después de mover el ratón
            delta_x = node_center_x - screen_center_x

            # Pausa entre iteraciones para hacer el movimiento más gradual
            time.sleep(0.1)

        # Verificar si se centró al final del ciclo
        if abs(delta_x) < 1:
            print("[SUCCESS] El nodo se centró con éxito.")
        else:
            print("[INFO] No se alcanzó el centro en el número de pasos especificado.")


    def move_to_position(self, target_x):
        """Mueve el ratón hasta la posición target_x en otro hilo."""
        thread = threading.Thread(target=self.align_camera_to_node, args=(target_x,), daemon=True)
        thread.start()

    def comprobar_orientacion(self, node_center_x, tolerance=5):
        """Mueve el ratón de forma gradual hacia el centro del nodo, comprobando si está centrado."""
        # Obtener las dimensiones de la pantalla (1920x1080 como ejemplo)
        center_x = 1920 // 2
        # Calcular la diferencia entre el centro de la pantalla y el centro del objeto
        delta_x = center_x - node_center_x
        # Si el objeto ya está suficientemente centrado, no mover el ratón
        return delta_x

    def _move_mouse(self, target_x):
        """Función en segundo plano que mueve el ratón hasta target_x."""
        step = 5 if target_x > 0 else -5  # Dirección del movimiento
        while abs(target_x) > 0:
            move = min(abs(target_x), abs(step)) * (1 if target_x > 0 else -1)
            self.ui.write(ecodes.EV_REL, ecodes.REL_X, move)
            self.ui.syn()
            target_x -= move  # Reducir la distancia restante
            time.sleep(0.05)  # Pausa para suavizar el movimiento
    def enviar_nodo(self, node_center_x):
        self.coord = None
        """Agrega un nodo a la cola para que la cámara se alinee."""
        self.coord = node_center_x

    def luchar(self, x):

        self.ui.write(ecodes.EV_KEY, ecodes.KEY_3, 1)
        self.ui.syn() 
        self.ui.write(ecodes.EV_KEY, ecodes.KEY_3, 0)
        self.ui.syn() 
        time.sleep(1)

    def controlar_tecla(self):
        """
        Mantiene la tecla presionada mientras 'moverse' sea True.
        Si 'moverse' es False, deja de presionar la tecla.
        """
        while True:
            if self.moverse:
                self.ui.write(ecodes.EV_KEY, ecodes.KEY_W, 1)  # Presionar "W"
                self.ui.syn()  # Mantiene la tecla presionada
                self.finw = True
            else:
                if self.finw:
                    self.ui.write(ecodes.EV_KEY, ecodes.KEY_W, 0)  # Soltar "W"
                    self.ui.syn()
                    self.finw = False
            """
            if self.luchando:
                print("LUCHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                self.ui.write(ecodes.EV_KEY, ecodes.KEY_3, 1)
                self.ui.syn() 
                self.ui.write(ecodes.EV_KEY, ecodes.KEY_3, 0)
                self.ui.syn() 
                time.sleep(1)
                self.finl = True
            else:
                if self.finl:
                    self.ui.write(ecodes.EV_KEY, ecodes.KEY_3, 0)
                    self.ui.syn() 
                    self.finl = False
            """
            time.sleep(0.1)

    def coger_nodo(self):
        self.ui.write(ecodes.EV_KEY, ecodes.KEY_E, 1)  # Presionar "W"
        self.ui.syn()
        self.ui.write(ecodes.EV_KEY, ecodes.KEY_E, 0)  # Soltar "W"
        self.ui.syn()  
             

    def cambiar_estado(self, estado):
        """
        Cambia el estado de la variable 'moverse' a True o False.
        :param estado: Estado booleano para activar o desactivar el movimiento.
        """
        self.moverse = estado


    def align_camera_to_node2(self, node_center_x):
        """Mueve el ratón de forma gradual hacia el centro del nodo, comprobando si está centrado."""
        # Obtener las dimensiones de la pantalla (1920x1080 como ejemplo)
        screen_center_x = 1920 // 2  # Centro de la pantalla en X

        # 🔍 Verificar si `node_center_x` tiene sentido
        if node_center_x < 0 or node_center_x > 1920:
            print(f"[ERROR] El nodo está fuera de la pantalla: {node_center_x}")
            return 0

        # 🔥 CORRECCIÓN: Asegurar que `delta_x` es correcto
        delta_x = screen_center_x - node_center_x  # CAMBIADO: ahora es centro - nodo



        # Determinar si es necesario mover el ratón
        if abs(delta_x) < 1:  # Si la diferencia es muy pequeña, no moverse

            return 0

        # Limitar el movimiento para evitar saltos grandes
        max_step = 80 # Máximo movimiento por iteración
        step = min(abs(delta_x), max_step)  # Asegurar que no se pase del delta_x real
        move_x = -step if delta_x > 0 else step  # 🔥 CAMBIO: invertir la dirección



        # Enviar el evento de movimiento del ratón
        try:
            self.ui.write(ecodes.EV_REL, ecodes.REL_X, move_x)
            self.ui.syn()

        except Exception as e:
            print(f"[ERROR] Ocurrió un error al mover el ratón: {e}")

        time.sleep(0.1)



    def align_camera_in_thread(self, node_center_x):
        """Llama a `align_camera_to_node2` en un hilo separado."""
        # Crear el hilo que ejecutará la función en segundo plano
        camera_thread = threading.Thread(target=self.align_camera_to_node2, args=(node_center_x,))
        
        # Iniciar el hilo
        camera_thread.start()
        
        # Retornar el hilo para que pueda ser monitoreado si es necesario
        return camera_thread


    def turn(self):
        self.ui.write(ecodes.EV_REL, ecodes.REL_X, -100)
        self.ui.syn()
        #time.sleep(0.1)  # Esperar un poco antes de continuar el movimiento


    def encarar(self, x, y):
        pass

    def close(self):
        self.ui.close()  # Cierra el dispositivo virtual


class Control():

    def __init__(self):

        self.control = TeclaControl()
        self.path=[]
        self.capturar = False
        self.salir = False
        self.listener_thread = threading.Thread(target=self._run)
        self.listener_thread.start()
        self.mapa = None
        self.gamepad = vg.VX360Gamepad()
        self.track_nodo = SeguimientoNodos()
        self.track_enemigo = SeguimientoNodos()
        self.tiempo_inicio = time.time()
        self.distancia_actual = 0
        self.distancia_anterior = float('inf')
        self.space_pulsado = False
        self.ultimo_tiempo_comprobacion = time.time()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.f6:
                self.capturar = not self.capturar
            elif key == keyboard.Key.f12:
                self.salir = True
        except AttributeError:
            pass

    def _on_release(self, key):
        # Implement any functionality needed on key release
        pass

    def guardar_imagen(self, imagen, tstamp, nuevo_ancho=1024, nuevo_alto=576):
        """
        Guarda la imagen después de redimensionarla.
        :param imagen: Imagen a guardar (numpy array).
        :param tstamp: Timestamp para el nombre del archivo.
        :param nuevo_ancho: El nuevo ancho de la imagen después de redimensionarla.
        :param nuevo_alto: El nuevo alto de la imagen después de redimensionarla.
        """
        if not os.path.exists("./train/capturas"):
            os.makedirs("./train/capturas")
        
        # Redimensionar la imagen a las dimensiones deseadas
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        
        # Guardar la imagen redimensionada
        with ThreadPoolExecutor() as executor:
            executor.submit(lambda: cv2.imwrite(f"./train/capturas/{tstamp}.jpg", imagen_redimensionada))

    def iniciar(self, detector: ColorModelInterface, segmentador: MaskGeneratorInterface):
        self.mapa = Mapa()

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while True:
                if self.salir:
                    print("Adiós!")
                    self.control.cambiar_estado(False)
                    self.control.close()
                    break

                if not self.capturar:
                    screenshot = sct.grab(monitor)
                    imagen = np.array(screenshot)
                    imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
                    detector.inferencia(imagen_bgr, dibujar=False)
                    segmentador.inferencia(imagen_bgr, dibujar=False)

                    path = []
                    path2 = []
                    enemy = False
                    node = False

                    self.actual = time.time()
                    if detector.detectar_negro_en_caja():
                        if not detector.confirmacion_nodo():
                            self.control.cambiar_estado(False)
                            self.control.coger_nodo()
                            tiempo_transcurrido = time.time() - self.actual
                            print(f"Tiempo transcurrido: {tiempo_transcurrido:.2f}s")
                            if time.time() - self.ultimo_tiempo_comprobacion >= 6:
                                self.control.toque()
                                self.distancia_anterior = self.distancia_actual  
                                self.ultimo_tiempo_comprobacion = time.time()
                            continue

                    #if detector.obtener_enemigos(): 
                        #enemy = True
                        #coordenadas = detector.obtener_coord_enemigos()
                        #path2 = self.mapa.run(segmentador.obtener_mascara_reducida(), 
                                              #detector.obtener_coord_jugador(),
                                              #coordenadas)

                    
                    coordenadas = detector.obtener_coord_nodos()
                    path = self.mapa.run(segmentador.obtener_mascara_reducida(), 
                                             detector.obtener_coord_jugador(),
                                             coordenadas)
                    self.mapa.mostrar_mapa()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break  

                    lucha = False
                    #if path2:  
                    if detector.es_pixel_rojo():
                        self.control.cambiar_estado(False)
                        self.control.luchar(True)
                        continue

                    if path:
                        self.distancia_actual = self.distancia_euclidiana(detector.obtener_coord_jugador(), path[1])
                        
                        # Verificar si han pasado al menos 3 segundos antes de comparar
                        if time.time() - self.ultimo_tiempo_comprobacion >= 3:
                            if self.distancia_anterior is not None:  # Asegurar que no sea la primera iteración
                                print(f"Distancia actual: {self.distancia_actual:.2f}")
                                print(f"Distancia anterior: {self.distancia_anterior:.2f}")

                                if self.distancia_actual >= self.distancia_anterior:
                                    print("Distancia no ha disminuido, pulsando Space...")
                                    self.control.ejecutar_secuencia_bloqueo()
                                    self.space_pulsado = True  

                            # Actualizar valores después de la comparación
                            self.distancia_anterior = self.distancia_actual  
                            self.ultimo_tiempo_comprobacion = time.time()  # Reiniciar el tiempo de la última comparación

                        """ HAY ALGÚN ERROR EN EL QUE PATH DEJA DE TENER VALORES """
                        if -50 < self.control.comprobar_orientacion(path[1][0]) < 50:
                            self.control.cambiar_estado(True)
                        else:
                            self.control.align_camera_in_thread(path[1][0])
                            self.control.cambiar_estado(True)
                        continue

                    else:
                        self.control.turn()
                        self.control.cambiar_estado(False)
                    
                    
                    time.sleep(0.05)
            cv2.destroyAllWindows()


    def distancia_euclidiana(self, punto1, punto2):
        # Calcula la distancia euclidiana entre dos puntos (x1, y1) y (x2, y2)
        return np.sqrt((punto2[0] - punto1[0]) ** 2 + (punto2[1] - punto1[1]) ** 2)

    def reset_space_pulsado(self):
        # Si quieres permitir pulsar Space nuevamente después de un tiempo o una condición
        self.space_pulsado = False
            

    def _run(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()
    



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
        
            # Si el objeto está a la izquierda del centro de la pantalla, mover a la derecha
        pyautogui.move(-delta_x, 0, 0.5)

        
                # -------------------------------
                # Comprobación de nodo fuera de la vista



if __name__ == "__main__":
    control = Control()
    det= ColorModel("./models/det_nodos_ESO.pt")
    seg = BinaryMaskGenerator("./models/best_seg_obs.pt")
    control.iniciar(det, seg)
    torch.cuda.empty_cache()