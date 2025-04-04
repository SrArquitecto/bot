import numpy as np
import cv2
import torch
from ultralytics import YOLO
from abc import ABC, abstractmethod
import os

class MaskGeneratorInterface(ABC):
    @abstractmethod
    def inferencia(self, imagen, conf=0.3, filtro=None, dibujar=False):
        pass
        
    @abstractmethod
    def obtener_mascara_reducida(self, w=350, h=197):
        pass

    @abstractmethod
    def obtener_mascara(self):
        pass

    @abstractmethod
    def obtener_resultados(self):
        pass

    @abstractmethod
    def obtener_mascara_vacia(self):
        pass
    

class BinaryMaskGenerator(MaskGeneratorInterface):
    def __init__(self, ruta_modelo, tamanio_imagen=(1080, 1920), output_dir="./train/mascaras/"):
        """
        Inicializa la clase con el tamaño de la imagen de entrada (por defecto 1920x1080) y el modelo YOLO.
        :param tamanio_imagen: (altura, ancho) de la imagen original.
        :param ruta_modelo: Ruta del modelo YOLO.
        """
        self.tamanio_imagen = tamanio_imagen
        self.modelo = YOLO(ruta_modelo)
        self.resultados = None
        self.image = None
        self.image_results = np.ones(self.tamanio_imagen, dtype=np.uint8) * 255
        self.mascara_vacia = np.ones(self.tamanio_imagen, dtype=np.uint8)
        self.mascara_invertida = None
        self.output_dir = output_dir

    def inferencia(self, imagen, conf=0.3, filtro=None, dibujar=False):
        """
        Realiza la inferencia usando el modelo YOLO sobre la imagen de entrada.
        :param imagen: Imagen de entrada en formato numpy array.
        :return: Resultados de la segmentación.
        """
        # Realizamos la inferencia con el modelo YOLO
        if filtro is not None:
            self.resultados = self.modelo(imagen, conf=conf, classes=filtro)  
        else:
            self.resultados = self.modelo(imagen, conf=conf)# Esta línea pasa la imagen al modelo y obtiene las predicciones
        #return resultados  # Suponiendo que el modelo devuelve un listado de resultados 

        self._generar_mascara(dibujar)

        # Iteramos sobre los resultados de segmentación
    def _generar_mascara(self, dibujar=False):
        """
        Genera una imagen binaria a partir de las detecciones de máscaras YOLO.
        :param detecciones: Resultados de segmentación del modelo YOLO.
        :param confianza: Umbral mínimo de confianza para incluir una máscara.
        :return: Imagen binaria en formato numpy array.
        """
        # Acumulador para las máscaras
        #self.mascaras = np.array([])
        mascara_acumulada = None  # Inicializar la variable
        
        # Iteramos sobre los resultados de segmentación
        if self.resultados is not None:
            for resultado in self.resultados:
                if hasattr(resultado, 'masks') and resultado.masks is not None:
                    masks_data = resultado.masks.data.cpu().numpy()  # Extraer las máscaras como NumPy array
                    for mask in masks_data:
                        # Asegurarse de que la máscara esté en formato binario (0 o 1)
                        mask_resized = (mask > 0).astype(np.uint8)
                        #np.append(self.mascaras, mask_resized)

                        # Sumar directamente con cv2.bitwise_or
                        if mascara_acumulada is None:
                            mascara_acumulada = mask_resized.copy()
                        else:
                            mascara_acumulada = cv2.bitwise_or(mascara_acumulada, mask_resized)

            # Asegurar que la máscara acumulada tenga valores binarios (0 o 1)
            if mascara_acumulada is not None:
                mascara_acumulada = np.clip(mascara_acumulada, 0, 1).astype(np.uint8)
                mascara_acumulada = cv2.resize(mascara_acumulada, (self.tamanio_imagen[1], self.tamanio_imagen[0]), interpolation=cv2.INTER_NEAREST)
            else:
                # Si no hay máscaras, asignar una máscara vacía (blanca)
                mascara_acumulada = np.zeros(self.tamanio_imagen, dtype=np.uint8)

        self.mascara_invertida = 1 - mascara_acumulada
        self._reduce_obstacle_area(5)
        if dibujar:
            self._mostrar_mascara()


    def _reduce_obstacle_area(self, reduction_percentage=20):
        """Reduce el área de los obstáculos en un porcentaje dado."""
        # Calcular el tamaño del kernel proporcionalmente a ambas dimensiones
        reduction_value = max(1, int((reduction_percentage / 100) * min(self.mascara_invertida.shape[:2]) * 0.1))
        
        # Crear un kernel más pequeño para mayor precisión
        kernel = np.ones((reduction_value, reduction_value), np.uint8)
        
        # Aplicar erosión con múltiples iteraciones en lugar de un kernel gigante
        iterations = max(1, int(reduction_percentage / 10))
        self.mascara_invertida = cv2.erode(self.mascara_invertida, kernel, iterations=iterations)

        
    def obtener_mascara_reducida(self, w=350, h=197):
        return cv2.resize(self.mascara_invertida, (w, h))

    def obtener_mascara_vacia(self):
        mask = 1 - self.mascara_vacia
        return mask * 255

    def obtener_resultados(self):
        return self.resultados

    def obtener_mascara(self):
        return self.mascara_invertida
        
    def _mostrar_mascara(self):
        """
        Muestra la máscara generada. Si no hay resultados, se mostrará una máscara blanca.
        """
        if self.mascara_invertida is not None:
            self.image_results = (self.mascara_invertida * 255).astype(np.uint8)
        #else:
        # Si no hay resultados, se muestra una máscara blanca
            #self.image_results = np.ones(self.tamanio_imagen, dtype=np.uint8) * 255

        cv2.imshow("Mascara", self.image_results)
