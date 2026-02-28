import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




recognition_result = None
def result_callback(result, output_image, timestamp_ms):
  global recognition_result
  recognition_result = result


model_path="gesture_recognizer.task"
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.GestureRecognizerOptions(
  base_options=base_options,
  running_mode=vision.RunningMode.LIVE_STREAM,
  num_hands=2,
  result_callback=result_callback
)


recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
  success, frame = cap.read()

  frame = cv2.resize(frame, (1200, 900))
  frame = cv2.flip(frame, 1)

  rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_img)
  
  timestamp= (time.time()*1000)
  
  
  
  cv2.imshow("WebCam", frame)

  if cv2.waitKey(1) & 0xff==ord("q"):
    break

cap.release()
cv2.destroyAllWindows()