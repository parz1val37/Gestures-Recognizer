import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

def Hand_Landmark_Connections():
  model_path = "hand_landmarker.task"

  # callback function
  detection_result = None
  def result_callback(result, output_image, timestamp_ms):
    nonlocal detection_result
    detection_result = result

  # options tuning
  base_options = python.BaseOptions(model_asset_path=model_path)

  options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=result_callback
    )
  # landmarker object made from the options
  landmarker = vision.HandLandmarker.create_from_options(options)

  try:
    cap = cv2.VideoCapture(0)
    pTime = 0
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      # resize frame
      frame = cv2.resize(frame, (1200, 900))
      # flip the frame horizontally
      frame = cv2.flip(frame, 1)
      # BGR(frame by cv2) to RGB (mediapipe accepts RGB image)
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

      timestamp = int(time.time()*1000)
      landmarker.detect_async(mp_image, timestamp)

      if detection_result and detection_result.hand_landmarks:
        h, w, _ = frame.shape

        for hand_landmarks in detection_result.hand_landmarks: #type: ignore
          # Land mark points
          for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (162, 217, 188), -1)

          # Connect Landmarks
          for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]

            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)

            cv2.line(frame, (x1, y1), (x2, y2), (148, 161, 68), 2)

      def display_FPS_on_frame(frame, position=(10, 50),font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(70, 70, 191), thickness=2, line_type=cv2.LINE_AA):
        # FPS calculation
        nonlocal pTime
        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime

        # Display FPS on frame

        cv2.putText(frame, f"FPS: {fps}", position, font, scale, color, thickness, line_type)

      display_FPS_on_frame(frame)

      cv2.imshow("Hand Tracking", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
  finally:
    cap.release()
    cv2.destroyAllWindows()



if __name__== "__main__":
  Hand_Landmark_Connections()

  '''
  My System Specs:
  -> This was completely processed on CPU, FPS was consistent on 30-32 FPS, with CPU utilization of (22-26)%
  CPU: AMD Ryzen 7 7435HS
  GPU: NVIDIA GeForce RTX 3050 Laptop GPU
  '''