import cv2
import time
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--modelPath', type=str, default='examples/model/facemask.pt', help='Model.pt path(s)')
parser.add_argument('--frameSize', type=int, default='640', help='frame size')
parser.add_argument('--useCuda', action='store_true', help='Option to use CUDA')
opt = parser.parse_args()
print(opt)

# Load YOLO model
print("importing YOLO")
from ultralytics import YOLO
model = YOLO("tugas_bayucaraka/best.pt")
if opt.useCuda:
    print("sending model to cuda")
    model.to('cuda')

# Open the webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_count = 0
total_fps = 0
avg_fps = 0

while cap.isOpened():
    start = time.time()

    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model.predict(source=frame, save=False, save_txt=False, conf=0.5, verbose=False)
    boxes = results[0].boxes
    names = model.names
    confidance, class_ids = boxes.conf, boxes.cls.int()
    rects = boxes.xyxy.int()
    objectCount = 0
    for ind in range(boxes.shape[0]):
        objectCount = objectCount+1
        # print("Rect:", names[class_ids[ind].item()], confidance[ind].item(), rects[ind].tolist())
        kotak = (rects[ind].tolist())
        cv2.rectangle(frame, (kotak[0], kotak[1]), (kotak[2], kotak[3]), (0, 255, 255), 2)
        text = names[class_ids[ind].item()] + ": " + "{:.2f}".format(confidance[ind].item())
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_width, text_height = text_size
        cv2.rectangle(frame, (kotak[0], kotak[1] - text_height - 10), (kotak[0] + text_width + 10, kotak[1]), (0, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (kotak[0] + 5, kotak[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    end = time.time()

    frame_count += 1
    fps = 1 / (end - start)
    total_fps = total_fps + fps
    avg_fps = total_fps / frame_count

    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)
    cv2.putText(frame, "FPS average: "+str(int(avg_fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
    cv2.putText(frame, "Frame size: "+str(frame_height)+"x"+str(frame_width), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()