import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



def Hand_Recognition():
  recognition_result = None
  def result_callback(result, output_image, timestamp_ms):
    nonlocal recognition_result
    recognition_result = result


  model_path="gesture_recognizer.task"
  base_options = python.BaseOptions(model_asset_path=model_path)

  options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=result_callback
  )


  with vision.GestureRecognizer.create_from_options(options) as recognizer:
    try:
      cap = cv2.VideoCapture(0)
      pTime=0
      while cap.isOpened():
        success, frame = cap.read()
        if not success:
          break

        # frame = cv2.resize(frame, (1200, 800))
        frame = cv2.flip(frame, 1)

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_img)

        timestamp= (time.time()*1000)

        # upscale frame (frame drops by 3-4 -depends on system)
        def Upscale_Frame(frame):
          # Scale image up by 2x using high-quality interpolation
          width = int(frame.shape[1] * 2)
          height = int(frame .shape[0] * 2)
          dim = (width, height)

          # INTER_CUBIC is slower but provides better results for upscaling
          resized_img = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
          return resized_img

        # frame = Upscale_Frame(frame)

        def display_FPS_on_frame(frame, position=(10, 50),font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(70, 70, 191), thickness=2, line_type=cv2.LINE_AA):
          # FPS calculation
          nonlocal pTime
          cTime = time.time()
          fps = int(1/(cTime-pTime))
          pTime = cTime

          # Display FPS on frame

          cv2.putText(frame, f"FPS: {fps}", position, font, scale, color, thickness, line_type)

        display_FPS_on_frame(frame)


        cv2.imshow("Gesture Recognizer", frame)

        if cv2.waitKey(1) & 0xff==ord("q"):
          break

    except Exception as e:
      print(f"ERROR: {e}")

    finally:
      cap.release()
      cv2.destroyAllWindows()



if __name__ == "__main__":
  Hand_Recognition()