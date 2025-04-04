import numpy as np
import cv2
from ultralytics import YOLO
import os
from abc import ABC, abstractmethod

# Definir la interfaz para el modelo YOLO
class ColorModelInterface(ABC):
    @abstractmethod
    def inferencia(self, imagen, conf=0.5, filtro=None, dibujar=False):
        """
        Realiza la inferencia sobre una imagen de entrada utilizando el modelo YOLO.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la inferencia.
        """
        pass
        
    @abstractmethod
    def obtener_resultados(self):
        pass

    @abstractmethod
    def obtener_coord_jugador(self):
        pass

    @abstractmethod
    def obtener_coord_nodos(self):
        pass


    @abstractmethod
    def obtener_nodos(self):
        pass

    @abstractmethod
    def obtener_nodo_mas_grande(self):
        pass

    @abstractmethod
    def obtener_pos_jugador(self):
        pass

    @abstractmethod
    def obtener_deteccion_mas_grande(self):
        pass

    @abstractmethod
    def detectar_negro_en_caja(self):
        pass

    @abstractmethod
    def obtener_enemigos(self):
        pass

    @abstractmethod
    def obtener_coord_enemigos(self):
        pass

    @abstractmethod
    def detectar_vida_baja(self):
        pass

    @abstractmethod
    def es_pixel_rojo(self):
        pass

    @abstractmethod
    def confirmacion_nodo(self):
        pass

# Implementación de la clase que realiza la inferencia y obtiene las detecciones
class ColorModel(ColorModelInterface):
    
    class_names_to_id = {
        "accion": 0,
        "jugador": 1,
        "nodo": 2
    }
    
    def __init__(self, ruta, output_dir="./train/detecciones/"):
        """
        Inicializa el modelo YOLO para detección de objetos.
        :param ruta: Ruta del modelo YOLO.
        """
        self.MIN_AREA = 100
        self.enemy = None
        self.enemigos= []
        self.coord_enemigos = []
        self.modelo = YOLO(ruta)
        self.resultados = None
        self.coord_nodos = []
        self.nodo_mas_grande = []
        self.deteccion_mas_grande = None
        self.coord_deteccion_mas_grande = []
        self.nodos = []
        self.mayor = None
        self.pos_jugador = None
        self.imagen = None
        self.imagen_resultados = None
        self.red_lower_limit = np.array([0, 0, 90], dtype=np.uint8)
        self.red_upper_limit = np.array([70, 90, 255], dtype=np.uint8)
        self.lower_limit = np.array([250, 250, 0], dtype=np.uint8)  # Valores de color magenta más bajos
        self.upper_limit = np.array([255, 255, 0], dtype=np.uint8)  # Valores de color magenta más altos

    def inferencia(self, imagen, conf=0.5, filtro=[1], dibujar=False):
        """
        Realiza la inferencia sobre una imagen de entrada utilizando el modelo YOLO.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la inferencia.
        """
        
        # Copiar la imagen original para trabajar sobre ella
        alto, ancho = imagen.shape[:2]

# Calcular el 10% del alto
        recorte = int(alto * 0.1)

# Recortar la imagen eliminando el 10% superior e inferior
        self.imagen_recoger = imagen.copy()
        self.imagen = imagen
        self.imagen_resultados = self.imagen.copy()

        # Verificar si la imagen tiene 4 canales (RGBA) y convertirla a BGR si es necesario
        if self.imagen.shape[-1] == 4:  # Si tiene 4 canales (RGBA)
            self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_RGBA2BGR)
        elif self.imagen.shape[-1] == 3:  # Si tiene 3 canales (BGR)
            # Si ya está en BGR, no es necesario convertir
            pass
        else:
            print("Error: La imagen tiene un formato no válido.")
            return
        recoger = None
        # Crear una máscara de colores dentro del rango de magenta
        mask = cv2.inRange(self.imagen, self.lower_limit, self.upper_limit)
        enemy_mask = cv2.inRange(self.imagen, self.red_lower_limit, self.red_upper_limit)
        # Detectar los contornos (grupos de píxeles magenta)
        self.resultados, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.enemy, _ = cv2.findContours(enemy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Copiar de nuevo la imagen para mostrar los resultados
        self.imagen_resultados = imagen.copy()

        self._calcular_detecciones()
        self._calcular_enemigos()

        if dibujar:
            self._dibujar_cajas()
            self._mostrar_imagen()

    def obtener_coord_jugador(self):
        return (957, 670)
    
    def obtener_enemigos(self):
        return self.enemigos
    
    def obtener_coord_enemigos(self):
        return self.coord_enemigos

    def obtener_coord_nodos(self):
        return self.coord_nodos

    def obtener_resultados(self):
        return self.resultados

    def obtener_nodos(self):
        return self.nodos

    def obtener_deteccion_mas_grande(self):
        return self.mayor

    def obtener_nodo_mas_grande(self):
        return self.coord_deteccion_mas_grande

    def obtener_pos_jugador(self):
        return self.pos_jugador

    def detectar_negro_en_caja(self):
        # Extraer la región de la imagen
        region = self.imagen_recoger[613:633, 1048:1070]
        # Verificar si hay negro puro (0,0,0) o blanco puro (255,255,255)
        return np.any(np.all(region == [0, 0, 0], axis=-1)) or np.any(np.all(region == [255, 255, 255], axis=-1))

    def detectar_vida_baja(self):
        # Extraer la región de la imagen
        region = self.imagen[1082:1094, 931:943]
        
        # Definir el rango de valores para los píxeles "blancos" (255, 255, 255)
        tolerancia = 55 # Tolerancia que puedes ajustar
        max_val = np.array([0, 255, 255])  # El valor máximo de "blanco"
        min_val = np.array([0, 255 - tolerancia, 255 - tolerancia])  # Rango mínimo con tolerancia
        
        # Crear una máscara que detecte si un píxel está dentro del rango "blanco"
        #mask = np.all(np.logical_and(region >= min_val, region <= max_val), axis=-1)
    
        # Verificar si hay al menos un píxel dentro del rango "blanco"
        # Si hay un píxel blanco, devolvemos False, si no hay, devolvemos True
        return np.any(np.all(region == [9, 205, 209], axis=-1)) or np.any(np.all(region == [12, 202, 207], axis=-1))

    
        
    def confirmacion_nodo(self):
        # Lista de tuplas con: (ruta de plantilla, región [y1:y2, x1:x2])
        plantillas_info = [
            ("template.png", (645, 666, 1083, 1180)),
            ("template2.png", (612, 628, 1083, 1135)),
            ("template3.png", (612, 629, 1083, 1158))
        ]

        for path, (y1, y2, x1, x2) in plantillas_info:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Error al cargar la plantilla: {path}")
                continue

            img = self.imagen.copy()
            region = img[y1:y2, x1:x2]

            if region.size == 0:
                print(f"Región vacía para {path}: {y1}:{y2}, {x1}:{x2}")
                continue

            screenshot_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)

            if np.max(result) >= 0.7:
                print(f"Coincidencia encontrada con: {path}")
                return True  # Si alguna plantilla coincide, devuelve True

        return False  # Si ninguna plantilla coincide, devuelve False
    
    
    def es_pixel_rojo(self):
        # Obtener las dimensiones de la imagen
        alto, ancho, _ = self.imagen.shape
        
        # Encontrar las coordenadas del píxel central
        x_centro = ancho // 2
        y_centro = alto // 2
        
        # Obtener el valor del píxel central
        pixel_central = self.imagen[y_centro, x_centro]
        
        # Definir el rango para ROJO en BGR
        rojo_min = np.array([0, 0, 150])    # Azul y Verde bajos, Rojo mínimo 150
        rojo_max = np.array([50, 50, 255])  # Azul y Verde hasta 100, Rojo máximo 255
        
        # Definir el rango para AMARILLO en BGR
        amarillo_min = np.array([0, 120, 120])   # Azul bajo, Verde y Rojo altos
        amarillo_max = np.array([60, 255, 255])  # Azul hasta 100, Verde y Rojo hasta 255
        
        # Verificar si el píxel está en el rango rojo
        es_rojo = np.all(pixel_central >= rojo_min) and np.all(pixel_central <= rojo_max)
        
        # Verificar si el píxel está en el rango amarillo
        es_amarillo = np.all(pixel_central >= amarillo_min) and np.all(pixel_central <= amarillo_max)
        
        # Retornar True si el píxel es rojo o amarillo
        return es_rojo or es_amarillo
            

    def _calcular_enemigos(self):
        self.enemigos = []
        self.coord_enemigos = []
        if self.enemy:
            for contour in self.enemy:
                if cv2.contourArea(contour) >= self.MIN_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    xc = x + w // 2
                    yc = y + h // 2
                    self.enemigos.append((x, y, w, h))
                    self.coord_enemigos.append((xc, yc))

    def _calcular_detecciones(self):
        self.nodos = []
        self.mayor = ()
        self.coord_deteccion_mas_grande=()
        self.coord_nodos = []
        mejor_puntuacion = float('-inf')
        self.deteccion_mas_grande = None
        self.pos_jugador = ()
        if self.resultados:

            # Centro de la pantalla (en un array para cálculos más rápidos con NumPy)
            centro_pantalla = np.array([960, 540])

            # Dimensiones de la pantalla (puedes ajustar si cambian)
            ancho_pantalla = 1920
            alto_pantalla = 1080

            # Dividir la pantalla en tercios
            tercio_izquierdo = ancho_pantalla // 3
            tercio_derecho = 2 * ancho_pantalla // 3

            if self.resultados:
                for contour in self.resultados:
                    x, y, w, h = cv2.boundingRect(contour)    
                    nodo_centro = np.array([x + w // 2, y + h // 2])
                    xc = x + w // 2
                    yc = y + h // 2
                    area = h * w
                            # Calcular la distancia euclidiana desde el centro de la pantalla
                    distancia = np.linalg.norm(nodo_centro - centro_pantalla)

                            # Ponderar la puntuación (más área y menos distancia)
                    puntuacion =  area - distancia * 0.5

                            # Penalizar los nodos que estén en los tercios laterales (izquierda o derecha)
                    if nodo_centro[0] < tercio_izquierdo or nodo_centro[0] > tercio_derecho:
                        puntuacion -= 100  # Penaliza con un valor adecuado

                    self.nodos.append((x, y, w, h))
                    self.coord_nodos.append((xc, yc))
                            # Establecer un umbral para evitar que el objeto cambie de uno a otro fácilmente
                    umbral_proximidad = 50  # Ajusta este valor según sea necesario

                            # Si la puntuación es mejor y el objeto no está demasiado cerca de otro objeto
                    if puntuacion > mejor_puntuacion and (self.deteccion_mas_grande is None or
                        np.linalg.norm(nodo_centro - np.array([self.deteccion_mas_grande[1], self.deteccion_mas_grande[2]])) > umbral_proximidad):
                        mejor_puntuacion = puntuacion
                        self.mayor = (x, y, w, h)
                        self.deteccion_mas_grande = (x, y, w, h)
                        self.coord_deteccion_mas_grande = (xc, yc)
                            


    def _dibujar_cajas(self, color=(0, 255, 0), grosor=2):
        """
        Dibuja las cajas delimitadoras y etiquetas sobre la imagen.
        :param imagen: Imagen sobre la que se dibujarán las cajas.
        :param color: Color de las cajas y etiquetas.
        :param grosor: Grosor de las cajas.
        :return: Imagen con las cajas dibujadas.
        """
        
        if self.imagen is not None:
            self.imagen_resultados = self.imagen.copy()
            if self.nodos:
                for nodo in self.nodos:  
                    if nodo == self.deteccion_mas_grande:
                        color = (0, 0, 255) 
                    else:
                        color = (0, 255, 0)
                    x, y, w, h = nodo
                    cv2.rectangle(self.imagen_resultados, (x, y), (x + w, y + h), color, 2)
            if self.enemigos:
               #print(f"Enemigos detectados: {self.enemigos}")
                for enemigo in self.enemigos:
                    x, y, w, h = enemigo
                    cv2.rectangle(self.imagen_resultados, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
        

    def _mostrar_imagen(self):
        """
        Muestra la imagen con las detecciones.
        """
        
        if self.imagen_resultados is not None:
            cv2.imshow("Detecciones", self.imagen_resultados)
            


