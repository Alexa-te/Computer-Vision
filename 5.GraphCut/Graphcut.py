import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- Runtime File Selection ---
Tk().withdraw()  # Hide the root tkinter window
image_path = askopenfilename(title="Select an Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])

if not image_path:
    print("No image selected.")
    exit()

# --- Read the Image ---
image = cv2.imread(image_path)

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Process the Image ---
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# --- Annotate and Display ---
if results.pose_landmarks:
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    # Show using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Pose Estimation")
    plt.axis('off')
    plt.show()
else:
    print("No pose landmarks detected in the image.")
