import cv2 as cv
from vision.detectors.hexagon_detector  import HexagonDetector

def main():

  captura= cv.VideoCapture(0, cv.CAP_V4L2)

  detector=HexagonDetector()

  while True:

    ret,frame=captura.read() #return status==ret

    if not ret:
      print('falha na captura')
      break


    detection_result = detector.detect(frame)
    cv.imshow('teste visao- deteccao hexagono',frame)


    if cv.waitKey(1) & 0xFF== ord('q'):
      break


  captura.release()
  cv.destroyAllWindows()

if __name__ == '__main__': 
  main()