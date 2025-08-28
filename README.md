 🎭 Real-Time Emotion Detector

This project demonstrates how to build a **real-time emotion detection system** using **Python, OpenCV, and a pre-trained deep learning model**. It captures your webcam feed, detects faces, and predicts emotions with percentage probabilities — all running locally on your computer.

---

 📚 Features

* Detects human faces using **Haarcascade Classifier**
* Predicts **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
* Displays a **real-time webcam feed** with emotion labels
* Shows **probability bars** for each detected emotion

---

 🛠 Prerequisites

To run this project, you’ll need:

* Python (3.10 recommended)
* Visual Studio Code (or any IDE)
* A working webcam
* Basic Python knowledge
* Internet (for installing dependencies)

---

 📂 Project Structure

```
Emotion_Detector_ML/
│
├── venv/                          # Virtual environment  
├── _mini_XCEPTION.102-0.66.hdf5   # Pretrained model file  
├── emotion_detector.py             # Main Python script  
├── requirements.txt                # List of dependencies  
```

---

 🪜 How It Works

1. **Face Detection**

   * Uses OpenCV’s Haarcascade to detect faces in real-time.

2. **Emotion Prediction**

   * Extracted face region is passed into a pre-trained **Mini-XCEPTION model**.
   * The model outputs probabilities for 7 different emotions.

3. **Visualization**

   * A bounding box and label (emotion) are drawn on the face.
   * A side panel shows colored bars representing the percentage probability of each emotion.

---

 🚀 Steps to Run

1. **Set Up Environment**

   * Create a virtual environment inside the project folder.
   * Activate the environment.

2. **Install Dependencies**

   * Install the required Python libraries from `requirements.txt`.

3. **Add Pretrained Model**

   * Place `_mini_XCEPTION.102-0.66.hdf5` in the project folder.

4. **Run the Program**

   * Execute the Python script:

     ```
     python emotion_detector.py
     ```

5. **Test in Real-Time**

   * The webcam opens and shows emotion predictions.
   * Press **Q** to quit the program.

---

 📊 Tech Stack

* **Python** — programming language
* **OpenCV** — face detection + video processing
* **TensorFlow / Keras** — deep learning model handling
* **NumPy** — numerical computations

---

 🎯 Use Cases

* Real-time **human-computer interaction**
* **Sentiment analysis** in video calls
* Building blocks for **AI-powered assistants**
* Fun beginner project for learning AI + CV

