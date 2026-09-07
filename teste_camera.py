import cv2 as cv
from hexagon_detector import HexagonDetector
from blob_detector import BlobDetection

def main():

  captura= cv.VideoCapture(0, cv.CAP_V4L2)

  detector_hex=HexagonDetector()
  detector_blob=BlobDetection()

  while True:

    ret,frame=captura.read() #return status==ret

    if not ret:
      print('falha na captura')
      break


    detections_hex = detector_hex.detect(frame)
    detections_blob= detector_blob.detect(frame)

    cv.imshow('teste visao- comparacao x blob ',frame)
    


    if cv.waitKey(1) & 0xFF== ord('q'):
      break


  captura.release()
  cv.destroyAllWindows()

if __name__ == '__main__': 
  main()