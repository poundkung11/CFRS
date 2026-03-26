import cv2
import os
import math
import time
import numpy as np
import threading
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
from scipy.spatial import distance as scipy_dist
from PIL import ImageFont, ImageDraw, Image

def put_thai_text(img, text, pos, color_bgr, font_size):
    try:
        font = ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("msgothic.ttc", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_ear(eye_landmarks):
    if len(eye_landmarks) < 6: return 0.0
    A = scipy_dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = scipy_dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = scipy_dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

class ClassroomMonitoringSystem:

    def __init__(self, known_faces_path='known_faces'):
        self.path = known_faces_path
        self.BLUR_MIN_THRESH = 40.0
        self.BLUR_MAX_THRESH = 60.0
        self.BLUR_FACE_RATIO = 800.0

        print(f"[{datetime.now()}] INFO: Booting Counting Service (YOLOv8n)...")
        self.yolo_model = YOLO("yolov8n.pt")  
        self.person_count = 0
        self.last_yolo_boxes = []
        self.yolo_skip_frames = 10
        self.frame_counter = 0

        print(f"[{datetime.now()}] INFO: Booting Behavior Service (MediaPipe Tasks)...")
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            import urllib.request
            print(f"[{datetime.now()}] WARN: Downloading MediaPipe Face Landmarker model...")
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", model_path)
            
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            output_facial_transformation_matrixes=True
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.EAR_THRESH = 0.18
        self.LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_name = {}
        self.name_to_label = {}
        self.is_lbph_trained = False
        self._load_and_train_database()

    def _load_and_train_database(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            return

        faces = []
        labels = []
        current_id = 0

        for filename in os.listdir(self.path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            name = os.path.splitext(filename)[0].split('_')[0].upper()
            
            if name not in self.name_to_label:
                self.name_to_label[name] = current_id
                self.label_to_name[current_id] = name
                current_id += 1
                
            label_id = self.name_to_label[name]
            img_path = os.path.join(self.path, filename)
            
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                mp_results = self.face_landmarker.detect(mp_image)
                
                if getattr(mp_results, 'face_landmarks', None):
                    face_landmarks = mp_results.face_landmarks[0]
                    img_h, img_w, _ = img.shape
                    x_coords = [lm.x * img_w for lm in face_landmarks]
                    y_coords = [lm.y * img_h for lm in face_landmarks]
                    
                    left = max(0, int(min(x_coords)))
                    top = max(0, int(min(y_coords)))
                    right = min(img_w - 1, int(max(x_coords)))
                    bottom = min(img_h - 1, int(max(y_coords)))
                    
                    face_crop = cv2.cvtColor(img[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
                    if face_crop.size > 0:
                        face_crop = cv2.resize(face_crop, (200, 200))
                        faces.append(face_crop)
                        labels.append(label_id)
            except Exception as e:
                pass
                
        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))
            self.is_lbph_trained = True

    def _check_blur_dynamic(self, rgb_image, top, right, bottom, left):
        face_width = right - left
        face_crop = rgb_image[top:bottom, left:right]
        if face_crop.size == 0: return False
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        dynamic_blur_thresh = min(self.BLUR_MAX_THRESH, max(self.BLUR_MIN_THRESH, self.BLUR_FACE_RATIO / max(face_width, 1.0)))
        return variance > dynamic_blur_thresh

    def estimate_head_pose(self, face_landmarks, img_w, img_h):
        model_points = np.array([
            (0.0, 0.0, 0.0),             
            (0.0, -330.0, -65.0),        
            (-225.0, 170.0, -135.0),     
            (225.0, 170.0, -135.0),      
            (-150.0, -150.0, -125.0),    
            (150.0, -150.0, -125.0)      
        ], dtype="double")
        pts_idxs = [1, 152, 33, 263, 61, 291]
        image_points = np.array([
            (face_landmarks[idx].x * img_w, face_landmarks[idx].y * img_h)
            for idx in pts_idxs
        ], dtype="double")

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4,1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success: return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        if pitch > 90: pitch -= 180
        elif pitch < -90: pitch += 180
        if yaw > 90: yaw -= 180
        elif yaw < -90: yaw += 180
        return pitch, yaw, roll

    def process_frame(self, frame, tracked_faces, current_time, resize_scale=1):
        self.frame_counter += 1
        img_h, img_w, _ = frame.shape
        
        if self.frame_counter % self.yolo_skip_frames == 1 or self.frame_counter == 1:
            try:
                # conf=0.55 / iou=0.4 heavily restricts duplicate overlapping YOLO boxes
                yolo_outs = self.yolo_model.predict(frame, classes=[0], conf=0.50, iou=0.45, verbose=False) 
                if len(yolo_outs) > 0 and len(yolo_outs[0].boxes) > 0:
                    self.last_yolo_boxes = yolo_outs[0].boxes.xyxy.cpu().numpy()
                else:
                    self.last_yolo_boxes = []
                self.person_count = len(self.last_yolo_boxes)
            except Exception as e:
                pass

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_results = self.face_landmarker.detect(mp_image)

        detections = []
        face_locations = []
        behaviors = []

        if getattr(mp_results, 'face_landmarks', None):
            for face_landmarks in mp_results.face_landmarks:
                x_coords = [lm.x * img_w for lm in face_landmarks]
                y_coords = [lm.y * img_h for lm in face_landmarks]
                left, top = int(min(x_coords)), int(min(y_coords))
                right, bottom = int(max(x_coords)), int(max(y_coords))
                
                left = max(0, left)
                top = max(0, top)
                right = min(img_w - 1, right)
                bottom = min(img_h - 1, bottom)

                face_width = right - left
                if face_width < 20 or top >= bottom or left >= right: 
                    continue

                face_locations.append((top, right, bottom, left))

                left_eye_pts = [(face_landmarks[i].x * img_w, face_landmarks[i].y * img_h) for i in self.LEFT_EYE_IDXS]
                right_eye_pts = [(face_landmarks[i].x * img_w, face_landmarks[i].y * img_h) for i in self.RIGHT_EYE_IDXS]
                avg_ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
                
                pitch, yaw, roll = self.estimate_head_pose(face_landmarks, img_w, img_h)
                
                eyes_closed = avg_ear < self.EAR_THRESH
                head_bad_posture = abs(yaw) > 40 or pitch > 30 or pitch < -30 or abs(roll) > 30
                state = "หลับ/เหม่อ" if (eyes_closed or head_bad_posture) else "ตั้งใจเรียน"
                
                pose_penalty = 15.0 if head_bad_posture else 0.0
                
                behaviors.append({
                    "bbox": {"Top": top, "Right": right, "Bottom": bottom, "Left": left},
                    "state": state,
                    "pose_penalty": pose_penalty,
                    "debug_text": f"EAR:{avg_ear:.2f} P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}"
                })

            for location, behavior in zip(face_locations, behaviors):
                top, right, bottom, left = location
                bbox = behavior["bbox"]
                
                if not self._check_blur_dynamic(rgb_frame, top, right, bottom, left):
                    detections.append({
                        "Name": "ภาพเบลอ", "Confidence": 0.0, "BoundingBox": bbox, "State": behavior["state"], "debug_text": behavior["debug_text"]
                    })
                    continue

                if not self.is_lbph_trained:
                    detections.append({
                        "Name": "คนแปลกหน้า", "Confidence": 0.0, "BoundingBox": bbox, "State": behavior["state"], "debug_text": behavior["debug_text"]
                    })
                    continue

                face_crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (200, 200))
                    label_id, distance = self.face_recognizer.predict(face_crop)
                    
                    dynamic_threshold = 85.0 - behavior["pose_penalty"]
                    
                    if distance < dynamic_threshold:
                        name = self.label_to_name.get(label_id, "คนแปลกหน้า")
                        conf = max(0.0, min(100.0, 100.0 - (distance / 1.5)))
                    else:
                        name = "คนแปลกหน้า"
                        conf = 0.0
                else:
                    name = "คนแปลกหน้า"
                    conf = 0.0

                detections.append({
                    "Name": name, "Confidence": round(conf, 2), "BoundingBox": bbox, "State": behavior["state"], "debug_text": behavior["debug_text"]
                })

        # --- Yolo Fallback Logic ---
        for yb in self.last_yolo_boxes:
            x1, y1, x2, y2 = map(int, yb[:4])
            has_face = False
            for loc in face_locations:
                top, right, bottom, left = loc
                if left <= x2 + 30 and right >= x1 - 30 and top <= y2 + 30 and bottom >= y1 - 30:
                    has_face = True
                    break
            
            if not has_face:
                inherited_name = "ไม่พบใบหน้า"
                for t_id, t_data in tracked_faces.items():
                    tb = t_data["bbox"]
                    # If Face Tracker is anywhere near this YOLO box, steal its identified name!
                    if tb["Left"] <= x2 + 60 and tb["Right"] >= x1 - 60 and tb["Top"] <= y2 + 60 and tb["Bottom"] >= y1 - 60:
                        if t_data["name"] not in ["คนแปลกหน้า", "ภาพเบลอ", "ไม่พบใบหน้า", "Unknown"]:
                            inherited_name = t_data["name"]
                            break
                            
                detections.append({
                    "Name": inherited_name, 
                    "Confidence": 0.0, 
                    "BoundingBox": {"Top": y1, "Right": x2, "Bottom": y2, "Left": x1},
                    "State": "ฟุบหลับ/หันหลัง",
                    "debug_text": "YOLO Body Fallback"
                })

        return detections, self.person_count

if __name__ == "__main__":
    service = ClassroomMonitoringSystem()
    cap = cv2.VideoCapture(0)
    
    tracked_faces = {}
    next_track_id = 0
    confirmed_names_db = set()
    tracker_lock = threading.Lock()

    CONFIRMATION_TIME = 1.0
    TIMEOUT = 1.0
    MAX_DISTANCE = 100

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        current_time = time.time()
        results, person_count = service.process_frame(frame, tracked_faces, current_time)
        
        with tracker_lock:
            keys_to_delete = [k for k, v in tracked_faces.items() if current_time - v["last_seen"] > TIMEOUT]
            for k in keys_to_delete: del tracked_faces[k]

            for res in results:
                bb = res["BoundingBox"]
                ai_name = res["Name"]
                conf = res["Confidence"]
                state = res["State"]
                
                cx = (bb["Left"] + bb["Right"]) // 2
                cy = (bb["Top"] + bb["Bottom"]) // 2

                best_match_id = None
                min_dist = MAX_DISTANCE

                for t_id, t_data in tracked_faces.items():
                    dist_tracked = math.hypot(cx - t_data["centroid"][0], cy - t_data["centroid"][1])
                    if dist_tracked < min_dist:
                        min_dist = dist_tracked
                        best_match_id = t_id

                if best_match_id is None:
                    tracked_faces[next_track_id] = {
                        "name": ai_name, "centroid": (cx, cy),
                        "first_seen": current_time, "last_seen": current_time,
                        "confirmed": False, "state": state, 
                        "debug_text": res.get("debug_text", ""),
                        "bbox": bb
                    }
                    next_track_id += 1
                else:
                    t_data = tracked_faces[best_match_id]
                    t_data["centroid"] = (cx, cy)
                    t_data["last_seen"] = current_time
                    t_data["state"] = state
                    t_data["debug_text"] = res.get("debug_text", "")
                    t_data["bbox"] = bb
                    if not t_data["confirmed"] and ai_name not in ["คนแปลกหน้า", "ภาพเบลอ", "ไม่พบใบหน้า"]:
                        t_data["name"] = ai_name
                    
                    elapsed_time = current_time - t_data["first_seen"]
                    if elapsed_time >= CONFIRMATION_TIME:
                        t_data["confirmed"] = True

            for t_id, t_data in tracked_faces.items():
                bb = t_data["bbox"]
                state = t_data["state"]
                display_name = t_data["name"]
                elapsed_time = current_time - t_data["first_seen"]
                is_confirmed = t_data["confirmed"]

                if state == "ฟุบหลับ/หันหลัง":
                    color = (0, 0, 255)
                    text = f"{display_name} - {state}"
                elif display_name == "ไม่พบใบหน้า":
                    color = (0, 0, 255)
                    text = f"ไม่พบใบหน้า - {state}"
                elif display_name == "ภาพเบลอ":
                    color = (0, 165, 255)
                    text = f"ภาพขยับ/เบลอ - {state}"
                elif display_name == "คนแปลกหน้า":
                    color = (0, 0, 255)
                    text = f"คนแปลกหน้า - {state}" 
                else:
                    if is_confirmed:
                        color = (0, 255, 0) if state == "ตั้งใจเรียน" else (0, 165, 255)
                        text = f"{display_name} (ยืนยันแล้ว) - {state}"
                        if display_name not in confirmed_names_db:
                            print(f">>> [API/Database] เช็คชื่อ: {display_name} เวลา: {datetime.now()}")
                            confirmed_names_db.add(display_name)
                    else:
                        color = (0, 255, 255)
                        countdown = max(0, CONFIRMATION_TIME - elapsed_time)
                        text = f"กำลังตรวจสอบ {display_name}... {countdown:.1f}วิ - {state}"

                cv2.rectangle(frame, (bb["Left"], bb["Top"]), (bb["Right"], bb["Bottom"]), color, 2)
                frame = put_thai_text(frame, text, (bb["Left"], max(0, bb["Top"] - 30)), color, 24)
                
                debug_text = t_data.get("debug_text", "")
                if debug_text:
                    frame = put_thai_text(frame, debug_text, (bb["Left"], bb["Bottom"] + 5), (0, 255, 255), 18)

        frame = put_thai_text(frame, f"จำนวนคนในห้องทั้งหมด: {person_count} คน", (20, 20), (255, 255, 0), 32)
        cv2.imshow('Ultimate Classroom Identity + Behavior', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()