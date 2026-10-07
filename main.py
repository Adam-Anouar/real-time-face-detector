import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

if face_cascade.empty():
    print("Error: Could not load cascade XML file.")
    raise SystemExit(1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    raise SystemExit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    h, w = frame.shape[:2]
    new_w = 600
    new_h = int(h * (new_w / w))
    resized_frame = cv2.resize(frame, (new_w, new_h))

    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    for (x, y, box_w, box_h) in faces:
        cv2.rectangle(
            resized_frame,
            (x, y),
            (x + box_w, y + box_h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            resized_frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Stream Detector", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()