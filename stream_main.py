import cv2
import time
import math
import threading
import queue
import requests
import json
import numpy as np
from datetime import datetime
 
# ─────────────────────────────────────────
#  [INTEGRATOR CONFIG]  แก้ค่าตรงนี้ก่อน
# ─────────────────────────────────────────
 
USE_IP_WEBCAM   = False                         # True = มือถือ / False = webcam ในเครื่อง
IPWEBCAM_URL    = "http://192.168.1.105:8080"  # <-- เปลี่ยนเป็น IP จาก IP Webcam app
CAMERA_INDEX    = 0                            # ใช้เมื่อ USE_IP_WEBCAM = False
 
BACKEND_URL     = "http://127.0.0.1:5000/api/result"  # URL ของ Backend (แก้ตาม Backend จริง)
BACKEND_ENABLED = False                         # False = ปิดการส่ง backend (debug mode)
 
FRAME_WIDTH     = 640                          # ความกว้างของภาพที่ประมวลผล
FRAME_HEIGHT    = 480
 
SHOW_DEBUG_INFO = True                         # แสดง FPS / Thread status บนจอ
WINDOW_TITLE    = "CFRS - Classroom Monitor"
 
# ─────────────────────────────────────────
#  นำเข้า AI Engine จาก main.py เดิม
# ─────────────────────────────────────────
from main import ClassroomMonitoringSystem, put_thai_text
 
 
# ══════════════════════════════════════════════════════════════
#  1. VIDEO STREAM MODULE  (คุมท่อรับภาพจากมือถือ/webcam)
# ══════════════════════════════════════════════════════════════
 
class VideoStream:
    """
    อ่านภาพจาก IP Webcam (RTSP/HTTP) หรือ Webcam ธรรมดา
    ทำงานใน Thread แยก ไม่บล็อก Main Loop
    """
 
    def __init__(self, use_ip_webcam: bool, ip_url: str, cam_index: int):
        self._lock   = threading.Lock()
        self._frame  = None
        self._running = False
        self._thread  = None
        self.fps_in   = 0.0
 
        if use_ip_webcam:
            # IP Webcam รองรับทั้ง HTTP MJPEG และ RTSP
            # ลอง HTTP ก่อน (เสถียรกว่าผ่าน Wi-Fi hotspot)
            mjpeg_url  = f"{ip_url}/video"
            rtsp_url   = f"rtsp://{ip_url.split('//')[1]}/h264_ulaw.sdp"
 
            print(f"[STREAM] ลอง HTTP MJPEG: {mjpeg_url}")
            cap = cv2.VideoCapture(mjpeg_url)
            if not cap.isOpened():
                print(f"[STREAM] HTTP ล้มเหลว → ลอง RTSP: {rtsp_url}")
                cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                raise RuntimeError(
                    f"[STREAM] ❌ เชื่อมต่อกล้องไม่ได้!\n"
                    f"  → ตรวจสอบ Wi-Fi / Hotspot\n"
                    f"  → มือถือและแล็ปท็อปต้องอยู่ใน Network เดียวกัน\n"
                    f"  → URL ที่ใช้: {mjpeg_url}"
                )
        else:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                raise RuntimeError(f"[STREAM] ❌ เปิด Webcam index={cam_index} ไม่ได้")
 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลด buffer เพื่อให้ภาพ realtime ที่สุด
 
        self._cap = cap
        print(f"[STREAM] ✅ เชื่อมต่อกล้องสำเร็จ")
 
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._reader_loop, name="StreamReader", daemon=True)
        self._thread.start()
 
    def _reader_loop(self):
        t_prev = time.time()
        frame_count = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                print("[STREAM] ⚠️  อ่านภาพไม่ได้ — รอ 0.5 วิ แล้วลองใหม่")
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame = frame
            frame_count += 1
            now = time.time()
            if now - t_prev >= 1.0:
                self.fps_in   = frame_count / (now - t_prev)
                frame_count   = 0
                t_prev        = now
 
    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
 
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._cap.release()
 
 
# ══════════════════════════════════════════════════════════════
#  2. AI WORKER THREADS  (รัน AI 3 ตัวพร้อมกัน)
# ══════════════════════════════════════════════════════════════
#
#  ทำไมต้อง Multi-Thread?
#  ─────────────────────────────────────────────────────────
#  AI แต่ละตัว (YOLO / MediaPipe / LBPH) ใช้เวลาไม่เท่ากัน
#  ถ้ารันตามลำดับ (sequential) จะทำให้ภาพกระตุกมาก
#  การแยก Thread ทำให้:
#    • Thread 0: อ่านภาพ (VideoStream) → เร็วมาก ไม่รอใคร
#    • Thread 1: AI Processing (ClassroomMonitoringSystem)
#               → รับ frame ล่าสุด ประมวลผล แล้วส่งผลไปคิว
#    • Thread 2: Sender → ส่ง result ไป Backend (ไม่บล็อก display)
#    • Main Thread: แค่วาดผลบนหน้าจอ → ภาพนิ่ง ไม่กระตุก
#
#  Pattern ที่ใช้: Producer-Consumer ผ่าน queue.Queue
# ─────────────────────────────────────────────────────────
 
class AIWorker:
    """
    Thread ที่คอยรับ frame → ส่ง ClassroomMonitoringSystem → ใส่ผลลง result_queue
    """
 
    def __init__(self, system: ClassroomMonitoringSystem, result_queue: queue.Queue):
        self._system       = system
        self._result_queue = result_queue
        self._frame_queue  = queue.Queue(maxsize=1)  # เก็บแค่ frame ล่าสุด
        self._running      = False
        self._thread       = None
        self._tracked      = {}          # ใช้ tracked_faces ร่วมกับ AI engine
        self._next_id      = 0
        self._lock         = threading.Lock()
        self.fps_ai        = 0.0
        self.last_error    = ""
 
    def submit_frame(self, frame):
        """ส่ง frame ใหม่เข้าคิว — ถ้าคิวเต็มให้ทิ้ง frame เก่าก่อน (ไม่สะสม lag)"""
        if not self._frame_queue.empty():
            try: self._frame_queue.get_nowait()
            except queue.Empty: pass
        try: self._frame_queue.put_nowait(frame)
        except queue.Full: pass
 
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._work_loop, name="AIWorker", daemon=True)
        self._thread.start()
 
    def _work_loop(self):
        t_prev = time.time()
        count  = 0
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
 
            try:
                current_time = time.time()
                with self._lock:
                    tracked_snapshot = dict(self._tracked)
 
                detections, person_count = self._system.process_frame(
                    frame, tracked_snapshot, current_time
                )
 
                # อัปเดต tracker (logic เดิมจาก main.py แต่ย้ายมาอยู่ใน thread)
                with self._lock:
                    self._update_tracker(detections, person_count, current_time)
                    result_payload = self._build_payload(person_count, current_time)
 
                # ส่งผลลงคิวสำหรับ display และ sender
                if not self._result_queue.full():
                    self._result_queue.put_nowait({
                        "tracked":      dict(self._tracked),
                        "person_count": person_count,
                        "payload":      result_payload,
                        "frame":        frame,
                    })
 
                count += 1
                now = time.time()
                if now - t_prev >= 1.0:
                    self.fps_ai = count / (now - t_prev)
                    count = 0
                    t_prev = now
 
            except Exception as e:
                self.last_error = str(e)
                print(f"[AI] ⚠️  Error: {e}")
 
    # ── Tracker logic (ย้ายมาจาก main.py ให้อยู่ใน thread) ──────────────
    CONFIRMATION_TIME = 1.2
    TIMEOUT           = 2.0
    MAX_DISTANCE      = 100
    DROWSY_HOLD_TIME  = 1.5
 
    def _update_tracker(self, detections, person_count, current_time):
        # ลบ track ที่หายไปนาน
        dead = [k for k, v in self._tracked.items()
                if current_time - v["last_seen"] > self.TIMEOUT]
        for k in dead: del self._tracked[k]
 
        used = set()
        for res in detections:
            bb       = res["BoundingBox"]
            ai_name  = res["Name"]
            raw_state= res["State"]
            cx = (bb["Left"] + bb["Right"])  // 2
            cy = (bb["Top"]  + bb["Bottom"]) // 2
 
            best_id, min_d = None, self.MAX_DISTANCE
            for tid, td in self._tracked.items():
                if tid in used: continue
                d = math.hypot(cx - td["centroid"][0], cy - td["centroid"][1])
                if d < min_d:
                    min_d, best_id = d, tid
 
            if best_id is None:
                self._tracked[self._next_id] = {
                    "name": ai_name, "centroid": (cx, cy),
                    "first_seen": current_time, "last_seen": current_time,
                    "confirmed": False, "state": "ตั้งใจเรียน",
                    "raw_state": raw_state,
                    "drowsy_since": current_time if raw_state in ["หลับ/เหม่อ", "ฟุบหลับ/หันหลัง"] else None,
                    "debug_text": res.get("debug_text", ""),
                    "bbox": bb,
                }
                used.add(self._next_id)
                self._next_id += 1
            else:
                used.add(best_id)
                td = self._tracked[best_id]
                td["centroid"] = (cx, cy)
                td["last_seen"] = current_time
                td["raw_state"] = raw_state
                td["debug_text"] = res.get("debug_text", "")
                td["bbox"] = bb
 
                if raw_state in ["หลับ/เหม่อ", "ฟุบหลับ/หันหลัง"]:
                    if td.get("drowsy_since") is None:
                        td["drowsy_since"] = current_time
                else:
                    td["drowsy_since"] = None
 
                if raw_state == "ฟุบหลับ/หันหลัง":
                    td["state"] = "ฟุบหลับ/หันหลัง"
                elif td.get("drowsy_since") and (current_time - td["drowsy_since"] >= self.DROWSY_HOLD_TIME):
                    td["state"] = "หลับ/เหม่อ"
                else:
                    td["state"] = "ตั้งใจเรียน"
 
                if not td["confirmed"] and ai_name not in ["คนแปลกหน้า", "ภาพเบลอ", "ไม่พบใบหน้า"]:
                    td["name"] = ai_name
 
                if current_time - td["first_seen"] >= self.CONFIRMATION_TIME:
                    td["confirmed"] = True
 
    def _build_payload(self, person_count, current_time) -> dict:
        """สร้าง Data Package ที่จะส่งให้ Backend"""
        students = []
        for tid, td in self._tracked.items():
            students.append({
                "track_id":  tid,
                "name":      td["name"],
                "state":     td["state"],
                "confirmed": td["confirmed"],
            })
        return {
            "timestamp":    datetime.now().isoformat(),
            "person_count": person_count,
            "students":     students,
        }
 
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
 
    def get_tracked(self) -> dict:
        with self._lock:
            return dict(self._tracked)
 
 
# ══════════════════════════════════════════════════════════════
#  3. BACKEND SENDER THREAD  (ส่งข้อมูลไป Backend ไม่กระทบ display)
# ══════════════════════════════════════════════════════════════
 
class BackendSender:
    """
    รับ payload จาก send_queue แล้ว POST ไป Backend ใน Thread แยก
    ถ้า Backend ล่ม จะ retry 3 ครั้ง แล้ว skip ไม่ให้ระบบค้าง
    """
 
    RETRY_MAX    = 3
    RETRY_DELAY  = 0.3  # วินาที
 
    def __init__(self, backend_url: str, enabled: bool):
        self._url      = backend_url
        self._enabled  = enabled
        self._queue    = queue.Queue(maxsize=30)
        self._running  = False
        self._thread   = None
        self.last_status = "ยังไม่ได้ส่ง"
        self._sent_names = set()  # ป้องกันส่งชื่อซ้ำซ้อน
 
    def enqueue(self, payload: dict):
        if not self._enabled: return
        if not self._queue.full():
            self._queue.put_nowait(payload)
 
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._send_loop, name="BackendSender", daemon=True)
        self._thread.start()
 
    def _send_loop(self):
        session = requests.Session()
        while self._running:
            try:
                payload = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
 
            # ─── ส่งเฉพาะชื่อที่ยืนยันแล้วและยังไม่เคยส่ง ───────────
            new_students = [
                s for s in payload.get("students", [])
                if s["confirmed"] and s["name"] not in [
                    "คนแปลกหน้า", "ภาพเบลอ", "ไม่พบใบหน้า"
                ] and s["name"] not in self._sent_names
            ]
            for s in new_students:
                self._sent_names.add(s["name"])
                print(f"[BACKEND] >>> เช็คชื่อ: {s['name']}  เวลา: {payload['timestamp']}")
 
            # ─── POST ไป Backend พร้อม retry ─────────────────────────
            for attempt in range(self.RETRY_MAX):
                try:
                    resp = session.post(
                        self._url,
                        json=payload,
                        timeout=2.0,
                    )
                    if resp.status_code == 200:
                        self.last_status = f"✅ OK [{datetime.now().strftime('%H:%M:%S')}]"
                    else:
                        self.last_status = f"⚠️ HTTP {resp.status_code}"
                    break
                except requests.exceptions.ConnectionError:
                    self.last_status = "❌ Backend ไม่ตอบสนอง"
                    if attempt < self.RETRY_MAX - 1:
                        time.sleep(self.RETRY_DELAY)
                except Exception as e:
                    self.last_status = f"❌ Error: {e}"
                    break
 
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
 
 
# ══════════════════════════════════════════════════════════════
#  4. NETWORK HEALTH CHECK  (ตรวจสอบ Wi-Fi/Hotspot ก่อนพรีเซนต์)
# ══════════════════════════════════════════════════════════════
 
def check_network(ip_webcam_url: str, backend_url: str):
    """
    ตรวจสอบการเชื่อมต่อก่อน start ระบบ
    พิมพ์สถานะออกมาให้ทราบล่วงหน้า
    """
    print("\n" + "═"*55)
    print("  🔌 ตรวจสอบ Network ก่อนเริ่มระบบ")
    print("═"*55)
 
    # ─── ตรวจ IP Webcam ───────────────────────────────────────
    webcam_check_url = f"{ip_webcam_url}/status.json"
    try:
        r = requests.get(webcam_check_url, timeout=3)
        if r.status_code == 200:
            print(f"  ✅ IP Webcam : {ip_webcam_url} — ออนไลน์")
        else:
            print(f"  ⚠️  IP Webcam : HTTP {r.status_code} — ตรวจสอบแอป")
    except Exception:
        print(f"  ❌ IP Webcam : {ip_webcam_url} — ไม่ตอบสนอง")
        print("       → มือถือและแล็ปท็อปอยู่ใน Wi-Fi เดียวกันไหม?")
        print("       → เปิด IP Webcam แล้วกด 'Start server' หรือยัง?")
 
    # ─── ตรวจ Backend ─────────────────────────────────────────
    try:
        r = requests.get(backend_url.replace("/api/result", "/health"), timeout=2)
        print(f"  ✅ Backend   : {backend_url} — ออนไลน์")
    except Exception:
        print(f"  ⚠️  Backend   : {backend_url} — ไม่ตอบสนอง (ระบบยังทำงานได้ แต่ไม่บันทึกผล)")
 
    print("═"*55 + "\n")
 
 
# ══════════════════════════════════════════════════════════════
#  5. DISPLAY / DRAW  (วาดผลบนหน้าจอ)
# ══════════════════════════════════════════════════════════════
 
def draw_results(frame, tracked_faces: dict, person_count: int,
                 fps_in: float, fps_ai: float, backend_status: str):
    """วาด bounding box, ชื่อ, สถานะ และ debug info บนภาพ"""
 
    # ─── วาด Bounding Box + ชื่อ ──────────────────────────────
    for tid, td in tracked_faces.items():
        bb    = td["bbox"]
        state = td["state"]
        name  = td["name"]
        confirmed  = td["confirmed"]
        first_seen = td["first_seen"]
        elapsed    = time.time() - first_seen
 
        CONFIRMATION_TIME = 1.2
 
        if state == "ฟุบหลับ/หันหลัง":
            color = (0, 0, 255)
            label = f"{name} - {state}"
        elif name == "ไม่พบใบหน้า":
            color = (0, 0, 255)
            label = f"ไม่พบใบหน้า - {state}"
        elif name == "ภาพเบลอ":
            color = (0, 165, 255)
            label = "ภาพขยับ/เบลอ"
        elif name == "คนแปลกหน้า":
            color = (0, 0, 255)
            label = f"คนแปลกหน้า - {state}"
        else:
            if confirmed:
                color = (0, 255, 0) if state == "ตั้งใจเรียน" else (0, 165, 255)
                label = f"{name} (ยืนยันแล้ว) - {state}"
            else:
                color = (0, 255, 255)
                countdown = max(0, CONFIRMATION_TIME - elapsed)
                label = f"กำลังตรวจสอบ {name}... {countdown:.1f}วิ"
 
        cv2.rectangle(frame, (bb["Left"], bb["Top"]), (bb["Right"], bb["Bottom"]), color, 2)
        frame = put_thai_text(frame, label, (bb["Left"], max(0, bb["Top"] - 30)), color, 22)
 
        debug = td.get("debug_text", "")
        if debug and SHOW_DEBUG_INFO:
            frame = put_thai_text(frame, debug, (bb["Left"], bb["Bottom"] + 5), (0, 255, 255), 16)
 
    # ─── แถบข้อมูล HUD บนภาพ ──────────────────────────────────
    frame = put_thai_text(frame,
        f"จำนวนคนในห้อง: {person_count} คน",
        (20, 20), (255, 255, 0), 30)
 
    if SHOW_DEBUG_INFO:
        frame = put_thai_text(frame,
            f"Stream FPS: {fps_in:.1f}  |  AI FPS: {fps_ai:.1f}",
            (20, 60), (180, 180, 180), 20)
        frame = put_thai_text(frame,
            f"Backend: {backend_status}",
            (20, 85), (180, 255, 180), 18)
 
    return frame
 
 
# ══════════════════════════════════════════════════════════════
#  6. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════
 
def main():
    print("\n" + "═"*55)
    print("  🎓 CFRS — Classroom Face Recognition System")
    print("  Role: System Integrator & Network")
    print("═"*55)
 
    # ── 6.1 ตรวจสอบ Network ──────────────────────────────────
    if USE_IP_WEBCAM:
        check_network(IPWEBCAM_URL, BACKEND_URL)
 
    # ── 6.2 โหลด AI Model ────────────────────────────────────
    print("[MAIN] กำลังโหลด AI Models... (อาจใช้เวลา 10-30 วิ)")
    system = ClassroomMonitoringSystem(known_faces_path="known_faces")
    print("[MAIN] ✅ AI Models โหลดเสร็จแล้ว")
 
    # ── 6.3 เปิด Video Stream ─────────────────────────────────
    stream = VideoStream(
        use_ip_webcam=USE_IP_WEBCAM,
        ip_url=IPWEBCAM_URL,
        cam_index=CAMERA_INDEX,
    )
    stream.start()
 
    # ── 6.4 เริ่ม AI Worker Thread ────────────────────────────
    result_queue = queue.Queue(maxsize=2)
    ai_worker    = AIWorker(system, result_queue)
    ai_worker.start()
 
    # ── 6.5 เริ่ม Backend Sender Thread ──────────────────────
    sender = BackendSender(backend_url=BACKEND_URL, enabled=BACKEND_ENABLED)
    sender.start()
 
    # ── 6.6 Main Loop (Display Only — ไม่ทำงานหนักที่นี่) ─────
    print("[MAIN] ✅ ระบบเริ่มทำงาน — กด 'q' เพื่อออก\n")
 
    latest_tracked = {}
    latest_count   = 0
 
    try:
        while True:
            # รับภาพ frame ล่าสุด
            frame = stream.read()
            if frame is None:
                time.sleep(0.02)
                continue
 
            # ส่ง frame ให้ AI Worker ประมวลผล (ไม่รอผล)
            ai_worker.submit_frame(frame.copy())
 
            # ดึงผลล่าสุดจาก AI (ถ้ามี)
            try:
                result = result_queue.get_nowait()
                latest_tracked = result["tracked"]
                latest_count   = result["person_count"]
                # ส่ง payload ไป Backend (non-blocking)
                sender.enqueue(result["payload"])
            except queue.Empty:
                pass  # ไม่มีผลใหม่ → ใช้ผลเดิม
 
            # วาดผลบนภาพ
            display = draw_results(
                frame.copy(),
                latest_tracked,
                latest_count,
                fps_in=stream.fps_in,
                fps_ai=ai_worker.fps_ai,
                backend_status=sender.last_status,
            )
 
            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[MAIN] กำลังปิดระบบ...")
                break
 
    finally:
        # ── 6.7 Graceful Shutdown ─────────────────────────────
        stream.stop()
        ai_worker.stop()
        sender.stop()
        cv2.destroyAllWindows()
        print("[MAIN] ✅ ปิดระบบเรียบร้อย")
 
 
if __name__ == "__main__":
    main()