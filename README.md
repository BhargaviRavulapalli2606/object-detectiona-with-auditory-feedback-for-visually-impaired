# object-detectiona-with-auditory-feedback-for-visually-impaired
Object detection systems with auditory feedback for visually impaired individuals leverage computer vision and deep learning to identify objects in real-time, then convert this information into spoken cues. 
import cv2
import numpy as np
import requests
import time
from ultralytics import YOLO
import pyttsx3
import queue
from threading import Thread
import random # Added for TTS variety

# --- Configuration ---
ESP32_URL = 'http://192.168.186.192/capture'  # !!! IMPORTANT: Update with your ESP32-CAM IP and endpoint !!!
YOLO_MODEL_PATH = "yolov8n.pt" # Consider yolov8n (nano) for speed, or yolov8s (small) for potentially better accuracy
CONFIDENCE_THRESHOLD = 0.45 # Minimum confidence score (0.0 to 1.0) to consider a detection
TTS_RATE = 160 # Adjusted TTS rate (experiment for preference)
SPEAK_INTERVAL_SECONDS = 4  # Min time between announcing the same object class (reduced slightly)

# --- Distance Thresholds (Pixel Area) ---
# These are heuristic and depend heavily on camera resolution, lens, and typical object distances.
# Tune these values based on your specific setup and observations.
DISTANCE_THRESHOLDS = {
    "very_close": 50000,      # Object likely occupies a large portion of the view
    "moderately_close": 20000 # Object is clearly visible but not dominating the view
    # Anything smaller is considered "further away"
}

# --- Direction Zones (Percentage of Frame Width) ---
DIRECTION_ZONES = {
    "far_left": 0.20,      # Center X < 20% of width
    "left": 0.40,          # Center X between 20% and 40%
    # Center is between 40% and 60%
    "right": 0.80,         # Center X between 60% and 80%
    "far_right": 1.0       # Center X > 80% of width
}

# --- Network and Timing ---
REQUEST_TIMEOUT_SECONDS = 5 # Timeout for fetching image from ESP32
LOOP_DELAY_SECONDS = 0.05 # Minimum delay between loop iterations (adjust for desired FPS vs CPU usage)

# --- Visualization ---
BOX_COLORS = {
    "very_close": (0, 0, 255),       # Red
    "moderately_close": (0, 255, 255), # Yellow
    "further_away": (0, 255, 0)      # Green
}
TEXT_COLOR = (255, 255, 255) # White

# --- Initialization ---

# Initialize YOLOv8 model
print("[INFO] Loading YOLOv8 model...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    class_names = model.model.names
    print(f"[INFO] YOLOv8 model ('{YOLO_MODEL_PATH}') loaded successfully with {len(class_names)} classes.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model from {YOLO_MODEL_PATH}: {e}")
    exit()

# Initialize TTS engine
print("[INFO] Initializing Text-to-Speech engine...")
engine = None
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', TTS_RATE)
    # Optional: List available voices and select one
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[0].id) # Change index to select different voice
    print(f"[INFO] TTS engine initialized with rate {TTS_RATE}.")
except Exception as e:
    print(f"[ERROR] Failed to initialize TTS engine: {e}")
    print("[WARNING] Continuing without Text-to-Speech functionality.")

# --- Asynchronous TTS Functionality ---
# NOTE: This pattern is common for non-blocking TTS.
# TODO: Add attribution if this specific async TTS structure was adapted from a particular source.
tts_queue = queue.Queue()
tts_thread_running = True
last_spoken_message = {} # Track last spoken message content to avoid rapid identical announcements

def speak_worker():
    """Worker thread function to process TTS requests from the queue."""
    print("[INFO] TTS worker thread started.")
    while tts_thread_running:
        try:
            text_to_speak = tts_queue.get(timeout=1) # Wait up to 1 sec for a message
            if engine:
                 print(f"[SPEAK] Announcing: '{text_to_speak}'")
                 engine.say(text_to_speak)
                 engine.runAndWait() # Blocks within this thread, but not the main loop
            else:
                 # Still print if TTS is disabled, for debugging
                 print(f"[SPEAK-DISABLED] Would have said: '{text_to_speak}'")
            tts_queue.task_done()
        except queue.Empty:
            # No message in queue, continue loop
            continue
        except Exception as e:
            print(f"[ERROR] Exception in TTS worker thread: {e}")
    print("[INFO] TTS worker thread finished.")

def speak_async(text):
    """Adds text to the TTS queue for asynchronous speaking, with basic deduplication."""
    global last_spoken_message
    current_time = time.time()
    # Avoid queueing the *exact same message* too quickly
    if text != last_spoken_message.get('text') or \
       current_time - last_spoken_message.get('time', 0) > 1.0: # Min 1 sec between identical messages
        tts_queue.put(text)
        last_spoken_message = {'text': text, 'time': current_time}

# Start TTS worker thread only if engine initialized
tts_thread = None
if engine:
    tts_thread = Thread(target=speak_worker, daemon=True)
    tts_thread.start()

# --- Network Session ---
session = requests.Session() # Use a session for potential connection reuse/efficiency

# --- Core Functions ---

def fetch_and_decode_image(url, session, timeout):
    """Fetches an image from the URL and decodes it using OpenCV."""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Decode as BGR

        if frame is None:
            print("[WARNING] Failed to decode image data.")
            return None
        return frame

    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out after {timeout} seconds fetching image from {url}.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to get image from {url}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Error processing image response or decoding: {e}")
        return None

def process_detections(results, frame_width, frame_height, class_names, confidence_thresh,
                       dist_thresholds, dir_zones, last_spoken_time, speak_interval):
    """Analyzes YOLO detection results, determines properties, and queues TTS messages."""
    detections_data = []
    current_time = time.time()

    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < confidence_thresh:
            continue

        class_id = int(box.cls[0])
        object_name = class_names.get(class_id, "Unknown Object") # Use a default

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        box_area = (x2 - x1) * (y2 - y1)

        # --- Determine Direction (using defined zones) ---
        relative_x = center_x / frame_width
        if relative_x < dir_zones["far_left"]:
            direction = "on your far left"
        elif relative_x < dir_zones["left"]:
            direction = "on your left"
        elif relative_x < dir_zones["right"]: # Implicitly covers center (40% to 60%)
             direction = "in front of you"
        elif relative_x < dir_zones["far_right"]:
            direction = "on your right"
        else:
            direction = "on your far right"

        # --- Determine Approximate Distance (using defined thresholds) ---
        # This is a heuristic based on apparent size in the frame.
        if box_area > dist_thresholds["very_close"]:
            distance = "very close"
        elif box_area > dist_thresholds["moderately_close"]:
            distance = "moderately close"
        else:
            distance = "further away"

        # --- Prepare data for visualization ---
        label = f"{object_name}: {confidence:.2f} ({distance})"
        color = BOX_COLORS.get(distance, (0, 255, 0)) # Default to green if somehow uncategorized

        detections_data.append({
            'box': (x1, y1, x2, y2),
            'label': label,
            'color': color,
            'class_name': object_name, # Needed for TTS logic
            'direction': direction,
            'distance': distance
        })

        # --- TTS Logic ---
        last_time = last_spoken_time.get(object_name, 0)
        if current_time - last_time > speak_interval:
            # Choose randomly from a few sentence structures
            tts_message_options = [
                f"{object_name} detected {distance} {direction}.",
                f"There is a {object_name} {distance}, {direction}.",
                f"I see a {object_name} {direction}, it appears {distance}."
            ]
            # Maybe add a more urgent phrasing for very close objects
            if distance == "very close":
                tts_message_options.append(f"Warning, {object_name} very close {direction}!")

            message = random.choice(tts_message_options)
            speak_async(message) # Use async speak
            last_spoken_time[object_name] = current_time # Update last spoken time for this class

    return detections_data

def draw_visualizations(frame, detections_data):
    """Draws bounding boxes and labels on the frame."""
    for data in detections_data:
        x1, y1, x2, y2 = data['box']
        label = data['label']
        color = data['color']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Calculate text size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 10, label_height) # Position label above box, ensuring it's within frame top

        # Draw filled rectangle background for label
        cv2.rectangle(frame, (x1, label_y - label_height - baseline), (x1 + label_width, label_y), color, cv2.FILLED)

        # Draw label text
        cv2.putText(frame, label, (x1, label_y - baseline // 2), # Adjusted baseline position
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return frame

# --- Main Loop ---
last_spoken_time = {} # Tracks the last time a specific *class* of object was announced
print("[INFO] Starting snapshot detection loop.")
print(f"[INFO] Fetching images from: {ESP32_URL}")
print("[INFO] Press 'q' in the OpenCV window or CTRL+C in console to quit.")

keep_running = True
try:
    while keep_running:
        start_time = time.time()

        # 1. Get Image
        frame_original = fetch_and_decode_image(ESP32_URL, session, REQUEST_TIMEOUT_SECONDS)
        if frame_original is None:
            time.sleep(1) # Wait a bit longer if fetching failed
            continue

        # 2. Preprocess Image (Resize for YOLO)
        # Standard YOLOv8 input size is often 640, but check model specifics if needed
        target_size = (640, 480)
        try:
             frame_resized = cv2.resize(frame_original, target_size)
             frame_height, frame_width, _ = frame_resized.shape
        except Exception as e:
             print(f"[ERROR] Failed to resize frame: {e}")
             continue

        # 3. Object Detection
        try:
            # verbose=False prevents YOLO from printing stats to console each time
            results = model(frame_resized, verbose=False)[0]
        except Exception as e:
            print(f"[ERROR] Error during YOLO inference: {e}")
            continue

        # 4. Process Detections & Generate TTS
        # This function now handles direction, distance, TTS queueing, and prepares drawing data
        processed_data = process_detections(results, frame_width, frame_height, class_names,
                                            CONFIDENCE_THRESHOLD, DISTANCE_THRESHOLDS, DIRECTION_ZONES,
                                            last_spoken_time, SPEAK_INTERVAL_SECONDS)

        # 5. Draw Visualizations
        frame_display = draw_visualizations(frame_resized, processed_data)

        # 6. Display Frame
        cv2.imshow("ESP32 Snapshot Detection", frame_display)

        # 7. Frame Rate Control and Quit Check
        elapsed_time = time.time() - start_time
        sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
        time.sleep(sleep_time)

        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed, initiating shutdown.")
            keep_running = False

except KeyboardInterrupt:
    print("\n[INFO] CTRL+C detected. Initiating shutdown.")
    keep_running = False # Ensure loop condition is false

finally:
    # --- Cleanup ---
    print("[INFO] Cleaning up resources...")

    # Stop the TTS worker thread gracefully
    tts_thread_running = False
    if tts_thread and tts_thread.is_alive():
        print("[INFO] Waiting for TTS thread to finish...")
        tts_thread.join(timeout=2) # Wait up to 2 seconds
        if tts_thread.is_alive():
            print("[WARNING] TTS thread did not finish cleanly.")

    # Close OpenCV windows
    cv2.destroyAllWindows()
    print("[INFO] OpenCV windows destroyed.")

    # Close network session (optional, good practice)
    session.close()
    print("[INFO] Network session closed.")

    print("[INFO] Application finished.")
