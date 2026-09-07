
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar
import yaml

# base class for all detectors

@dataclass
class ModelsDetection:
    model: str
    class_id: int
    class_label: str
    conf: float
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    center: Tuple[int, int]
    rel_error: Tuple[float, float]
    center_dist: Optional[Tuple[float, float]] = None
    distance_m: Optional[float] = None

def calculate_relative_error(
    target_center: Tuple[int, int],
    frame_shape: Tuple[int, int, int],
) -> Tuple[float, float]:
    """Calcula o erro normalizado do alvo em relação ao centro da imagem."""
    height, width = frame_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    error_x = (target_center[0] - center_x) / center_x
    error_y = (center_y - target_center[1]) / center_y
    return round(error_x, 6), round(error_y, 6)





class BlobDetection:

    DETECTION_TYPE = "BLOB"

    def __init__(self,config_path="config/parameters.yaml"):
        with open (config_path,'r') as f:
            self.config=yaml.safe_load(f)
        blob_cfg= self.config['blob_filters']
        #Blob Detection Params
        params=cv.SimpleBlobDetector_Params()

        params.minThreshold = blob_cfg['minThreshold']
        params.maxThreshold = blob_cfg['maxThreshold']
        params.thresholdStep = blob_cfg['thresholdStep']

        params.filterByColor=True
        params.blobColor=255

        params.filterByConvexity= True
        params.minConvexity=blob_cfg['minConvexity']
        params.maxConvexity= blob_cfg['maxConvexity']

        params.filterByCircularity=True
        params.minCircularity=blob_cfg['minCircularity']
        params.maxCircularity=blob_cfg['maxCircularity']

        params.filterByArea=True
        params.minArea=blob_cfg['minArea']
        params.maxArea=blob_cfg['maxArea']

        self.detector=cv.SimpleBlobDetector_create(params)
        
    ''' 
    Implemente aqui as funções auxiliares necessárias para a detecção.
    
    Em Python, métodos iniciados com "_" são considerados protegidos e
    destinam-se ao uso interno da classe.
    exemplo: def _nome_do_metodo(self, frame: np.ndarray): 
    '''
    def _rgb_to_hsv (self,img_bgr):
        return cv.cvtColor(img_bgr,cv.COLOR_BGR2HSV)

    def _red_mask(self,img_hsv):

        claro1 = np.array(self.config['hsv_mask']['lower_red1'])
        escuro1 = np.array(self.config['hsv_mask']['upper_red1'])
    
        claro2 = np.array(self.config['hsv_mask']['lower_red2'])
        escuro2 = np.array(self.config['hsv_mask']['upper_red2'])    
    
        mascara1 = cv.inRange(img_hsv, claro1, escuro1)
        mascara2 = cv.inRange(img_hsv, claro2, escuro2)

        mascara = cv.bitwise_or(mascara1, mascara2)
        
        return mascara  
        

    def _filter_by_area(self,mascara):
        contornos, _ = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
        return contornos



    def detect(self, frame: np.ndarray) -> list[ModelsDetection]:
        detections: list[ModelsDetection] = []
        
        frame_hsv = self._rgb_to_hsv(frame)
        mascara = self._red_mask(frame_hsv)
        keypoints= self.detector.detect(mascara)

        for kp in keypoints:
            center_x=int(kp.pt[0])
            center_y=int(kp.pt[1])
            diameter=int(kp.size)

            w=diameter
            h=diameter
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Desenha a caixa delimitadora no frame (azul para diferenciar de vértices)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(frame, "BLOB", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            rel_err = calculate_relative_error((center_x, center_y), frame.shape)
            
            detections.append(
              ModelsDetection(
                model=self.DETECTION_TYPE,
                class_id=0,
                class_label="hexagon",
                conf=1.0, # valor fixo
                bbox=(x,y,w,h), 
                center=(center_x,center_y), 
                rel_error=rel_err 
                    )
                )

        ''' 
        Implemente aqui a lógica principal do detector.
        
        O método deve receber um frame da câmera e retornar uma lista
        de objetos ModelsDetection contendo as informações do(s)
        hexágono(s) detectado(s).
        '''

        # Ordena as detecções pela distância ao quadrado até o centro da imagem
        # (rel_error é o erro em x/y normalizado), colocando primeiro o hexágono
        # mais próximo do centro do frame.
        detections.sort(key=lambda m: m.rel_error[0]**2 + m.rel_error[1]**2)

        return detections
