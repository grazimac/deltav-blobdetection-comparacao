
import cv2 as cv
import numpy as np
from core.utils import load_yaml
from .base_detector import BaseDetector, ModelsDetection
from vision.utils.transformations import calculate_relative_error

class HexagonDetector(BaseDetector):

    DETECTION_TYPE = "HEXAGON"

    def __init__(self): 
        self.config = load_yaml("vision/config/hexagon.yaml")

    ''' 
    Implemente aqui as funções auxiliares necessárias para a detecção.
    
    Em Python, métodos iniciados com "_" são considerados protegidos e
    destinam-se ao uso interno da classe.
    exemplo: def _nome_do_metodo(self, frame: np.ndarray): 
    '''
    def _rgb_to_hsv (self,img_hsv):
        return cv.cvtColor(img_hsv,cv.COLOR_BGR2HSV)

    def _red_mask(self,img_hsv):

        claro1 = np.array([0, 150, 100])
        escuro1 = np.array([10, 255, 255])
    
        claro2 = np.array([170, 150, 100])
        escuro2 = np.array([180, 255, 255])    
    
        mascara1 = cv.inRange(img_hsv, claro1, escuro1)
        mascara2 = cv.inRange(img_hsv, claro2, escuro2)

        mascara = cv.bitwise_or(mascara1, mascara2)
        
        return mascara  
        

    def _find_red_contours(self,mascara):
        contornos, _ = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
        return contornos

    def _vertices (self, contornos):
        for contorno in contornos:
            perimetro = cv.arcLength(contorno, True)
            vertices = cv.approxPolyDP(contorno,0.25*perimetro,True) #o número é a tolerância, quanto menor,mais rígido e acentua mais os detalhes
            return vertices


    def detect(self, frame: np.ndarray) -> list[ModelsDetection]:
        detections: list[ModelsDetection] = []
        
        frame_hsv = self._rgb_to_hsv(frame)
        mascara = self._red_mask(frame_hsv)
        contornos =  self._find_red_contours(mascara)
        vertices = self._vertices(contornos)


        for contorno in contornos:
            #para de identificar coisas bem pequenas (menor que 500pixels)
            if cv.contourArea(contorno)<500:
                continue

            perimetro = cv.arcLength(contorno, True)
            vertices = cv.approxPolyDP(contorno,0.02*perimetro,True)
            
            if len(vertices) == 6:
                x,y,w,h = cv.boundingRect(contorno)
              #  cv.drawContours(frame, contornos, -1, (0, 255, 0), 1)
                cv.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
                print(f'hexágono confirmado, x={x}, y={y}, w={w}, h={h}')

                center_x= int(x+w/2)
                center_y=int(y+h/2)
                rel_err= calculate_relative_error((center_x,center_y),frame.shape)



                detections.append(
                    ModelsDetection(
                        model=self.DETECTION_TYPE,
                        class_id=0,
                        class_label="hexagon",
                        conf=1.0, # valor fixo
                        bbox=(x,y,w,h), #TODO: preencher
                        center=(center_x,center_y), #TODO: preencher
                        rel_error=rel_err #TODO: preencher - importe a função calculate_relative_error do arquivo em vision/utils/transformations.py para calcular o erro relativo
                    )
                )





        #cv.imshow("frame",frame_hsv)
        #cv.imshow("red_mask",mascara)
        #cv.imshow("contornos",frame)     
        #cv.waitKey(1)
        
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
