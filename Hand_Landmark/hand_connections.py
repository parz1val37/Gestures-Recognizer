import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

def Hand_Landmark_Connections():
  model_path = "Hand_Landmark/hand_landmarker.task"

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
      # frame = cv2.resize(frame, (1200, 900))
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

          # Collect z values for normalization
          z_values = [lm.z for lm in hand_landmarks]
          z_min = min(z_values)
          z_max = max(z_values)

          # Avoid division by zero
          z_range = z_max - z_min if z_max - z_min != 0 else 1

          # Draw landmarks (BLACK circles)
          for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

          # Draw depth-based grey connections
          for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]

            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)

            # Average depth of the connection
            avg_z = (start.z + end.z) / 2

            # Normalize depth → 0 to 1
            norm_z = (avg_z - z_min) / z_range

            # Map to grey intensity (closer = lighter)
            grey = int(255 * (1 - norm_z*1.2))

            cv2.line(frame, (x1, y1), (x2, y2), (grey, grey, grey), 2)

      # upscale frame (frame drops by 3-4 -depends on system)
      def Upscale_and_Resize_Frame(frame):
        # Scale image up by 2x using high-quality interpolation
        width = int(frame.shape[1] * 2)
        height = int(frame .shape[0] * 2)
        dim = (width, height)

        # INTER_CUBIC is slower but provides better results for upscaling
        resized_img = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
        return resized_img
      frame = Upscale_and_Resize_Frame(frame)

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