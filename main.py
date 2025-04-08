from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import requests



# Scaling factor: meters per pixel (adjust this based on calibration)
SCALING_FACTOR = 0.05  # Example: 1 pixel = 0.05 meters

# Initialize variables
start_time = time.time()
oldclass = ''
prev_position = None
flag = True

# Load the video and model
cap = cv2.VideoCapture("Drone.mp4")
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("best.pt")
classnames = ["Drone"]

def send_telegram_msg(message):
    TOKEN = "7614649225:AAEJNW0In97l_pBOKdxZd9ry0bRHdkx26NU"
    chat_id = "5341914866"  # Replace with the actual chat ID
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    response = requests.get(url)
    return response.json()

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2  # Center of the bounding box

            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.65:
                cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Drone detection message
                if flag:
                    send_telegram_msg(f'{classnames[cls]} Detected...')
                    oldclass = classnames[cls]
                    flag = False

                # Calculate speed and direction
                if prev_position:
                    dx = cx - prev_position[0]
                    dy = cy - prev_position[1]
                    distance_pixels = math.sqrt(dx**2 + dy**2)
                    elapsed_time = time.time() - start_time

                    if elapsed_time > 0:
                        # Convert speed to meters per second
                        speed_mps = (distance_pixels * SCALING_FACTOR) / elapsed_time

                        # Determine direction
                        direction = ""
                        if abs(dx) > abs(dy):
                            direction = "right" if dx > 0 else "left"
                        else:
                            direction = "down" if dy > 0 else "up"

                        # Send speed and direction update
                        send_telegram_msg(
                            f"Drone Detected! Speed: {speed_mps:.2f} m/s, Direction: {direction}"
                        )
                        start_time = time.time()  # Reset the timer

                # Update previous position
                prev_position = (cx, cy)

    cv2.imshow('Drone', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

