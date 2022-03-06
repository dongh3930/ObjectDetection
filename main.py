# import necessary packages
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

cap = cv2.VideoCapture('video1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    # apply object detection (물체 검출)
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov4-tiny')

    # draw bounding box over detected objects (검출된 물체 가장자리에 바운딩 박스 그리기)
    out = draw_bbox(frame, bbox, label, conf, write_conf=True)

    cv2.imshow('result', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()