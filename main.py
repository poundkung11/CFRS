import face_recognition
import cv2
import os
import math
import time
import numpy as np
import threading
from datetime import datetime

class ClassroomFacialRecognitionService:

    def __init__(self, known_faces_path='known_faces'):
        self.path = known_faces_path
        self.known_db = {}
        
        #Magic Numbers
        self.WEIGHT_BEST = 0.7
        self.WEIGHT_SECOND = 0.3
        self.CONFIDENCE_K = 12
        self.MIN_CONFIDENCE = 60.0
        
        self.BLUR_MIN_THRESH = 40.0
        self.BLUR_MAX_THRESH = 60.0
        self.BLUR_FACE_RATIO = 800.0

        self._load_and_encode_database()

    def _load_and_encode_database(self):
        """Indexing ด้วย num_jitters=5 พร้อม Error Handling"""
        print(f"[{datetime.now()}] INFO: Booting Identity Service...")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print(f"[{datetime.now()}] WARN: Directory created. Please add images to '{self.path}'.")
            return

        for filename in os.listdir(self.path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            name = os.path.splitext(filename)[0].split('_')[0].upper()
            img_path = os.path.join(self.path, filename)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[{datetime.now()}] WARN: Cannot read {filename}. Skipping...")
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_img, num_jitters=5)

                if len(encodings) > 0:
                    if name not in self.known_db:
                        self.known_db[name] = []
                    self.known_db[name].append(encodings[0])
                    print(f"  -> SUCCESS: Learned face from {filename} (High-Quality Jitter)")
                else:
                    print(f"  -> WARN: No face detected in {filename}")
            except Exception as e:
                print(f"[{datetime.now()}] ERROR: Failed to process {filename}: {e}")
        
        print(f"[{datetime.now()}] INFO: Ready! Loaded {len(self.known_db)} identities.")

    def _check_blur_dynamic(self, rgb_image, face_loc):
        top, right, bottom, left = face_loc
        face_width = right - left
        face_crop = rgb_image[top:bottom, left:right]
        if face_crop.size == 0: return False
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        dynamic_blur_thresh = min(self.BLUR_MAX_THRESH, max(self.BLUR_MIN_THRESH, self.BLUR_FACE_RATIO / max(face_width, 1.0)))

        return variance > dynamic_blur_thresh

    def _get_pose_penalty(self, face_landmarks):
        """[แก้ไขที่ 2] รับ Landmarks ที่คำนวณแบบ Batch มาแล้ว ไม่ต้องคำนวณใหม่ทีละหน้า"""
        if not face_landmarks: return 0.0
        
        lm = face_landmarks
        if 'left_eye' in lm and 'right_eye' in lm and 'nose_bridge' in lm:
            left_eye_center = np.mean(lm['left_eye'], axis=0)
            right_eye_center = np.mean(lm['right_eye'], axis=0)
            nose_center = np.mean(lm['nose_bridge'], axis=0)
            
            dist_left = np.linalg.norm(left_eye_center - nose_center)
            dist_right = np.linalg.norm(right_eye_center - nose_center)

            ratio = min(dist_left, dist_right) / max(dist_left, dist_right + 1e-6)
            if ratio < 0.4: return 0.08
            if ratio < 0.6: return 0.04
        return 0.0

    def get_dynamic_threshold_and_margin(self, face_width, pose_penalty):
        face_width = max(20, min(100, face_width))
        t = (face_width - 20) / (100 - 20)
        base_thresh = 0.60 - (0.10 * t)
        margin = 0.08 - (0.06 * t)
        pose_penalty = max(0.0, min(0.1, pose_penalty))
        final_thresh = base_thresh - pose_penalty
        final_thresh = max(0.45, min(0.65, final_thresh))

        return round(final_thresh, 3), round(margin, 3)

    def _get_weighted_distance(self, known_encodings, target_encoding):
        dists = face_recognition.face_distance(known_encodings, target_encoding)
        if len(dists) == 1:
            return dists[0]
        
        sorted_dists = np.sort(dists)
        weighted_dist = (sorted_dists[0] * self.WEIGHT_BEST) + (sorted_dists[1] * self.WEIGHT_SECOND)
        return weighted_dist

    def _calculate_confidence(self, distance, threshold):
        confidence = 1 / (1 + math.exp(self.CONFIDENCE_K * (distance - threshold)))
        return round(max(1.0, min(99.9, confidence * 100)), 2)

    def process_frame(self, frame, resize_scale=1):
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations, num_jitters=2)
        all_face_landmarks = face_recognition.face_landmarks(rgb_small, face_locations)

        detections = []

        for idx, (encoding, face_loc) in enumerate(zip(face_encodings, face_locations)):
            top, right, bottom, left = face_loc
            face_width = right - left
            scale_back = 1.0 / resize_scale
            y1, x2, y2, x1 = [int(v * scale_back) for v in face_loc]
            bbox = {"Top": y1, "Right": x2, "Bottom": y2, "Left": x1}

            if not self._check_blur_dynamic(rgb_small, face_loc):
                detections.append({"Name": "Moving/Blur", "Confidence": 0.0, "BoundingBox": bbox})
                continue

            current_landmarks = all_face_landmarks[idx] if idx < len(all_face_landmarks) else None
            pose_penalty = self._get_pose_penalty(current_landmarks)
            threshold, required_margin = self.get_dynamic_threshold_and_margin(face_width, pose_penalty)

            person_distances = {}
            for name, known_encodings in self.known_db.items():
                person_distances[name] = self._get_weighted_distance(known_encodings, encoding)

            sorted_persons = sorted(person_distances.items(), key=lambda x: x[1])

            name = "Unknown"
            conf = 0.0

            if len(sorted_persons) > 0:
                best_match_name, best_dist = sorted_persons[0]
                
                actual_margin = 1.0
                if len(sorted_persons) > 1:
                    actual_margin = sorted_persons[1][1] - best_dist

                conf = self._calculate_confidence(best_dist, threshold)

                if best_dist <= threshold and actual_margin > required_margin and conf >= self.MIN_CONFIDENCE:
                    name = best_match_name
                else:
                    name = "Unknown"

            detections.append({
                "Name": name,
                "Confidence": conf,
                "BoundingBox": bbox
            })

        return detections


if __name__ == "__main__":
    service = ClassroomFacialRecognitionService()
    cap = cv2.VideoCapture(0)

    tracked_faces = {}
    next_track_id = 0
    
    confirmed_names_db = set()

    tracker_lock = threading.Lock()

    CONFIRMATION_TIME = 5.0
    TIMEOUT = 10.0
    MAX_DISTANCE = 50

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        results = service.process_frame(frame)

        # ใช้งาน Thread Lock เมื่อมีการเข้าถึงหรือเขียนทับตัวแปรแชร์ร่วม
        with tracker_lock:
            # ล้าง Tracker ที่หลุดเฟรมไปเกิน TIMEOUT
            keys_to_delete = [k for k, v in tracked_faces.items() if current_time - v["last_seen"] > TIMEOUT]
            for k in keys_to_delete:
                del tracked_faces[k]

            for res in results:
                bb = res["BoundingBox"]
                ai_name = res["Name"]
                conf = res["Confidence"]
                
                cx = (bb["Left"] + bb["Right"]) // 2
                cy = (bb["Top"] + bb["Bottom"]) // 2

                best_match_id = None
                min_dist = MAX_DISTANCE

                for t_id, t_data in tracked_faces.items():
                    dist = math.hypot(cx - t_data["centroid"][0], cy - t_data["centroid"][1])
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = t_id

                is_tracking = False
                is_confirmed = False
                display_name = ai_name
                elapsed_time = 0.0

                if best_match_id is None:
                    if ai_name not in ["Unknown", "Moving/Blur"]:
                        tracked_faces[next_track_id] = {
                            "name": ai_name,
                            "centroid": (cx, cy),
                            "first_seen": current_time,
                            "last_seen": current_time,
                            "confirmed": False
                        }
                        display_name = ai_name
                        is_tracking = True
                        next_track_id += 1
                else:
                    t_data = tracked_faces[best_match_id]
                    t_data["centroid"] = (cx, cy)
                    t_data["last_seen"] = current_time
                    
                    # ถ้าชื่อยังไม่คอนเฟิร์ม แต่เฟรมนี้ระบบจำได้ชัดเจน ให้อัปเดตชื่อใน Tracker
                    if not t_data["confirmed"] and ai_name not in ["Unknown", "Moving/Blur"]:
                        t_data["name"] = ai_name
                    
                    elapsed_time = current_time - t_data["first_seen"]
                    if elapsed_time >= CONFIRMATION_TIME:
                        t_data["confirmed"] = True
                        display_name = t_data["name"]
                        is_confirmed = True
                        is_tracking = True
                    else:
                        display_name = t_data["name"]
                        is_tracking = True

                # Rendering
                if not is_tracking:
                    if display_name == "Moving/Blur":
                        color = (0, 165, 255)
                        text = "Moving/Blur"
                    else:
                        color = (0, 0, 255)
                        text = "Unknown" 
                else:
                    if is_confirmed:
                        color = (0, 255, 0)
                        text = f"{display_name} (Verified {conf}%)"
                        
                        # ตรวจสอบกับ confirmed_names_db 
                        if display_name not in confirmed_names_db:
                            print(f">>> [API/Database] เช็คชื่อ: {display_name} เวลา: {datetime.now()}")
                            confirmed_names_db.add(display_name)
                    else:
                        color = (0, 255, 255)
                        countdown = max(0, CONFIRMATION_TIME - elapsed_time)
                        text = f"Verifying {display_name}... {countdown:.1f}s"

                cv2.rectangle(frame, (bb["Left"], bb["Top"]), (bb["Right"], bb["Bottom"]), color, 2)
                cv2.putText(frame, text, (bb["Left"], bb["Top"] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Ultimate Classroom Identity (Tracker Lock)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()