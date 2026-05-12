from ultralytics import YOLO
import cv2

# Load model
model = YOLO('model/best_fixed.pt')

# Class names (in correct order after fix)
CLASS_NAMES = ['Bike', 'Helmet_Off', 'Helmet_On', 'No_helmet', 'Rider']

def detect_image(image_path):
    results = model.predict(
        source=image_path,
        conf=0.25,
        iou=0.45,
        show_labels=True,
        show_conf=True
    )
    # Plot and show
    annotated = results[0].plot()
    cv2.imshow('Traffic Violation Detection', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return results

def detect_video(video_path):
    results = model.predict(
        source=video_path,
        conf=0.25,
        iou=0.45,
        show=True,          # live window
        save=True,          # saves output video
        save_dir='output/'
    )
    return results

def detect_webcam():
    results = model.predict(
        source=0,           # 0 = default webcam
        conf=0.25,
        show=True,
        stream=True
    )
    for r in results:
        pass  # keep stream alive

# --- Run ---
if __name__ == '__main__':
    # Change to your use case:
    detect_image('test.jpg')
    # detect_video('traffic.mp4')
    # detect_webcam()