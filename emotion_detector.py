from keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0), (200,200,200)]

# Load Haar cascade
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    canvas = np.zeros((400, 300, 3), dtype="uint8")  # for percentage bar UI

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(roi)[0]
            emotion_idx = np.argmax(prediction)
            label = emotion_labels[emotion_idx]

            # Draw face box
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            # Bold label above face
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,255,0), 3, cv2.LINE_AA)

            # Draw bar graph with bold text
            for i, (emotion, prob) in enumerate(zip(emotion_labels, prediction)):
                bar_x = 10
                bar_y = 40 + i*45
                bar_width = int(prob * 250)

                # Background bar
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 250, bar_y + 30), (50, 50, 50), -1)

                # Filled bar
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), colors[i], -1)

                text = f"{emotion}: {int(prob*100)}%"

                # Black outline for readability
                cv2.putText(canvas, text, (bar_x + 10, bar_y + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)

                # Main colored text
                cv2.putText(canvas, text, (bar_x + 10, bar_y + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2, cv2.LINE_AA)

    # Combine main frame and canvas
    frame_resized = cv2.resize(frame, (600, 400))
    canvas_resized = cv2.resize(canvas, (300, 400))
    combined = np.hstack((frame_resized, canvas_resized))

    cv2.imshow('Emotion Detector with Bold UI', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
