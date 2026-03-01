import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

      while cap.isOpened():
        success, frame = cap.read()
        if not success:
          break

        frame = cv2.resize(frame, (1200, 900))
        frame = cv2.flip(frame, 1)

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_img)

        timestamp= int((time.time()*1000))
        recognizer.recognize_async(mp_image, timestamp)

        h, w, _ = frame.shape
        if latest_result and latest_result.hand_landmarks:
          for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):

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
              (0, 255, 0),2)

            # Get gesture for this hand
            if latest_result.gestures and len(latest_result.gestures) > idx:
              top_gesture = latest_result.gestures[idx][0]
              gesture_name = top_gesture.category_name
              score = round(top_gesture.score, 2)

              label = f"{gesture_name} ({score})"

              # Put text at bottom-right corner of box
              text_size, _ = cv2.getTextSize(label,
                cv2.FONT_HERSHEY_SIMPLEX,0.7,2)

              text_x = x_max - text_size[0]
              text_y = y_max + text_size[1]

              # Prevent overflow
              text_y = min(h - 10, text_y)

              cv2.putText(
              frame,label,(text_x, text_y),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.7,(0, 255, 0),2)






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