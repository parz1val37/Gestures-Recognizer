import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections



def Gesture_Recognizer():
  # callback function requires for LIVE_STREAM Mode
  latest_result = None
  def result_callback(result, mp_image, timestamp_ms):
    nonlocal latest_result
    latest_result = result

  model_path="gesture_recognizer.task"
  base_options = python.BaseOptions(model_asset_path=model_path)

  options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=2
  )

  with vision.GestureRecognizer.create_from_options(options) as recognizer:
    try:
      cap = cv2.VideoCapture(0)
      pTime=0
      while cap.isOpened():
        success, frame = cap.read()
        if not success:
          break

        frame = cv2.resize(frame, (1300, 900))
        frame = cv2.flip(frame, 1)

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_img)

        timestamp= int((time.time()*1000))
        recognizer.recognize_async(mp_image, timestamp)

        h, w, _ = frame.shape

        if latest_result and latest_result.hand_landmarks:
          def Show_Hand_detected():
            text="Hand Detected"
            (_, _), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            x = 10
            y = h - baseline - 10
            cv2.putText(frame, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
          Show_Hand_detected()
          for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
            #------------*- Rectangular Box -*------

            # Convert normalized landmarks to pixel coords
            x_coords = [int(lm.x * w) for lm in hand_landmarks]
            y_coords = [int(lm.y * h) for lm in hand_landmarks]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add small padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Draw bounding box
            cv2.rectangle(
              frame,
              (x_min, y_min),
              (x_max, y_max),
              (67, 119, 254),2)
            
            #---------*- HandLandmark Connections and Circle landmarkpoints-*-------
            
            # Draw landmarks (BLACK circles)
            for lm in hand_landmarks:
              cx, cy = int(lm.x * w), int(lm.y * h)
              cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)            
            
            # Collect z values for normalization
            z_values = [lm.z for lm in hand_landmarks]
            z_min = min(z_values)
            z_max = max(z_values)

            # Avoid division by zero
            z_range = z_max - z_min if z_max - z_min != 0 else 1
            
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

            #------ Overlaying gesture name on frame ---*----
            # After the gesture is recognized, label it and display on frame
            def display_Labeled_gesture(gesture_name: str, score):
              label = f"{gesture_name} ({score})"
              # Put text at bottom-right corner of box
              text_size, _ = cv2.getTextSize(label,
                cv2.FONT_HERSHEY_SIMPLEX,0.7,2)

              text_x = x_max - text_size[0]
              text_y = y_max + text_size[1]

              # Prevent overflow
              text_y = min(h - 10, text_y+12)

              cv2.putText(
                frame,label,(text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(240, 248, 247),2)
            
            #-------Custom Gestures------
            def is_One(hand_landmarks):
              index_up = hand_landmarks[8].y < hand_landmarks[6].y
              middle_down = hand_landmarks[12].y > hand_landmarks[10].y
              ring_down = hand_landmarks[16].y > hand_landmarks[14].y
              pinky_down = hand_landmarks[20].y > hand_landmarks[18].y

              return index_up and middle_down and ring_down and pinky_down
            
            def is_Three(hand_landmarks):
              index_up = hand_landmarks[8].y < hand_landmarks[6].y
              middle_up = hand_landmarks[12].y < hand_landmarks[10].y
              ring_up = hand_landmarks[16].y < hand_landmarks[14].y
              pinky_down = hand_landmarks[20].y > hand_landmarks[19].y

              return index_up and middle_up and ring_up and pinky_down

            def is_Four(hand_landmarks):
              index_up = hand_landmarks[8].y < hand_landmarks[6].y
              middle_up = hand_landmarks[12].y < hand_landmarks[10].y
              ring_up = hand_landmarks[16].y < hand_landmarks[14].y
              pinky_up = hand_landmarks[20].y < hand_landmarks[18].y
              thumb_down = (hand_landmarks[4].y > hand_landmarks[3].y) or (hand_landmarks[4].x>hand_landmarks[3].x)

              return index_up and middle_up and ring_up and pinky_up and thumb_down
            
            
            if is_One(hand_landmarks):
              gesture_name="One"
              score=0.8
              display_Labeled_gesture(gesture_name, score)
            
            elif is_Three(hand_landmarks):
              gesture_name="Three"
              score=0.77
              display_Labeled_gesture(gesture_name, score)
            
            elif is_Four(hand_landmarks):
              gesture_name="Four"
              score=0.83
              display_Labeled_gesture(gesture_name, score)

            # Get gesture for this hand
            elif latest_result.gestures and len(latest_result.gestures) > idx:              
              top_gesture = latest_result.gestures[idx][0]
              gesture_name = top_gesture.category_name
              score = round(top_gesture.score, 2)

              display_Labeled_gesture(gesture_name, score)

        else:
          # Hand not detected
          def Show_No_Hand_detected():
            text="No Hand Detected"
            (_, _), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            x = 10
            y = h - baseline - 10
            cv2.putText(frame, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
          Show_No_Hand_detected()

        def display_FPS_on_frame(frame, position=(10, 50),font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(70, 70, 191), thickness=2, line_type=cv2.LINE_AA):
          # FPS calculation
          nonlocal pTime
          cTime = time.time()
          fps = int(1/(cTime-pTime))
          pTime = cTime

          # Display FPS on frame
          cv2.putText(frame, f"FPS: {fps}", position, font, scale, color, thickness, line_type)
        display_FPS_on_frame(frame)


        cv2.imshow("GESTURE RECOGNIZER", frame)

        if cv2.waitKey(1) & 0xff==ord("q"):
          break

    except Exception as e:
      print(f"ERROR: {e}")

    finally:
      cap.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  Gesture_Recognizer()