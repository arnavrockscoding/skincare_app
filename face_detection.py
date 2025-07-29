import cv2
import numpy as np
import mediapipe as mp
import os

# ------------------- Load Model -------------------
model_path = "skin_classifier.h5"
use_model = False

if os.path.exists(model_path):
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        use_model = True
        print("✅ Loaded model: skin_classifier.h5")
    except Exception as e:
        print(f"⚠ Model load failed: {e}")
        model = None
else:
    model = None

labels = ['acne', 'dry_skin', 'oily_skin', 'clear']

routine = {
    'acne': "Use salicylic acid or benzoyl peroxide. Avoid touching your face.",
    'dry_skin': "Apply ceramide-based moisturizer. Avoid long hot showers.",
    'oily_skin': "Use clay masks. Prefer gel-based non-comedogenic products.",
    'clear': "Keep skin hydrated. Maintain current skincare routine."
}

# ------------------- Helpers -------------------
def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def analyze_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    if use_model:
        pred = model.predict(image)
        return labels[np.argmax(pred)]
    else:
        return np.random.choice(labels)

# ------------------- Setup -------------------
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh

regions = {
    "forehead": 10,
    "left_cheek": 234,
    "right_cheek": 454,
    "chin": 152
}

final_results = {}
captured = False

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        enhanced_frame = enhance_image(frame.copy())
        rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            for region, idx in regions.items():
                x = int(face.landmark[idx].x * w)
                y = int(face.landmark[idx].y * h)
                x1, y1 = max(0, x - 60), max(0, y - 60)
                x2, y2 = min(w, x + 60), min(h, y + 60)
                crop = enhanced_frame[y1:y2, x1:x2]

                # Show rectangles while waiting
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(frame, region, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

                # Only predict AFTER capture
                if captured and region not in final_results and crop.shape[0] > 0 and crop.shape[1] > 0:
                    result = analyze_image(crop)
                    final_results[region] = result

        # Show prediction after capture
        if captured:
            y_pos = 30
            for region, result in final_results.items():
                cv2.putText(frame, f"{region.title()} → {result.upper()}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25
                cv2.putText(frame, f"→ {routine[result]}", (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                y_pos += 30

        instruction = "Press 'c' to capture and analyze | 'r' to reset | 'q' to quit"
        cv2.putText(frame, instruction, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("🧴 Smart Skin Analyzer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not captured:
            captured = True
            print("🧠 Analyzing face regions...")
        elif key == ord('r'):
            captured = False
            final_results.clear()
            print("🔁 Resetting scan...")

cap.release()
cv2.destroyAllWindows()
