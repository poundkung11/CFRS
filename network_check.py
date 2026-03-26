import cv2
import time
import socket
import requests
import statistics
 
# ── แก้ค่าตรงนี้ตามจริง ──────────────────────────────────
IPWEBCAM_URL = "http://192.168.1.105:8080"   # URL จากแอป IP Webcam
BACKEND_URL  = "http://127.0.0.1:5000"
# ─────────────────────────────────────────────────────────
 
 
def ping_host(host: str, port: int, timeout: float = 2.0) -> float:
    """วัด latency ไปยัง host:port (ms)  คืนค่า -1 ถ้าเชื่อมไม่ได้"""
    try:
        start = time.time()
        with socket.create_connection((host, port), timeout=timeout):
            return (time.time() - start) * 1000
    except Exception:
        return -1.0
 
 
def test_stream(url: str, num_frames: int = 10):
    """ดึง frame N ภาพ วัด latency และ drop rate"""
    cap = cv2.VideoCapture(f"{url}/video")
    if not cap.isOpened():
        return None, 1.0  # latency=None, drop=100%
 
    latencies = []
    drops = 0
    for _ in range(num_frames):
        t0 = time.time()
        ret, _ = cap.read()
        dt = (time.time() - t0) * 1000
        if ret:
            latencies.append(dt)
        else:
            drops += 1
 
    cap.release()
    avg_lat = statistics.mean(latencies) if latencies else None
    drop_rate = drops / num_frames
    return avg_lat, drop_rate
 
 
def main():
    print("\n" + "═"*60)
    print("  🔍 CFRS Network Health Check — ก่อนพรีเซนต์")
    print("═"*60)
 
    host_part = IPWEBCAM_URL.split("//")[-1]
    host, port_str = host_part.split(":") if ":" in host_part else (host_part, "80")
    port = int(port_str)
 
    # ─── 1. TCP Ping ──────────────────────────────────────────
    print(f"\n[1] TCP Ping ไป IP Webcam ({host}:{port})")
    lat = ping_host(host, port)
    if lat < 0:
        print(f"  ❌ เชื่อมต่อไม่ได้!")
        print("     → ตรวจว่ามือถือ + แล็ปท็อปอยู่ใน Wi-Fi/Hotspot เดียวกัน")
        print("     → ตรวจว่าแอป IP Webcam เปิดอยู่และกด 'Start server'")
        print("     → ตรวจ Firewall ของ Windows (ปิดชั่วคราวได้)")
    elif lat < 20:
        print(f"  ✅ Latency: {lat:.1f} ms — ดีมาก")
    elif lat < 60:
        print(f"  ⚠️  Latency: {lat:.1f} ms — ใช้ได้ แต่อาจกระตุกเล็กน้อย")
    else:
        print(f"  ❌ Latency: {lat:.1f} ms — สูงมาก!")
        print("     → เปลี่ยนใช้ Hotspot จากมือถือแทน Wi-Fi สาธารณะ")
        print("     → ลองให้มือถือและแล็ปท็อปอยู่ใกล้กัน")
 
    # ─── 2. Stream Quality ────────────────────────────────────
    print(f"\n[2] ทดสอบ Stream (ดึง 10 ภาพ)")
    avg_lat, drop_rate = test_stream(IPWEBCAM_URL, num_frames=10)
 
    if avg_lat is None:
        print("  ❌ ดึง Stream ไม่ได้เลย — ตรวจ URL และการเชื่อมต่อ")
    else:
        print(f"  ⏱  Latency เฉลี่ยต่อภาพ : {avg_lat:.0f} ms")
        print(f"  📉 Drop rate           : {drop_rate*100:.0f}%")
        if drop_rate == 0 and avg_lat < 200:
            print("  ✅ Stream คุณภาพดีพร้อมพรีเซนต์!")
        elif drop_rate > 0.2:
            print("  ❌ Drop rate สูงเกินไป — ภาพจะกระตุกมาก")
            print("     → ลด Resolution ในแอป IP Webcam (360p หรือ 480p)")
            print("     → ใช้ Hotspot แทน Wi-Fi")
        else:
            print("  ⚠️  Stream ใช้ได้ — ลอง tune ค่า resolution ถ้าอยากดีขึ้น")
 
    # ─── 3. Backend ───────────────────────────────────────────
    print(f"\n[3] ตรวจ Backend ({BACKEND_URL})")
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        print(f"  ✅ Backend ออนไลน์ (HTTP {r.status_code})")
    except Exception:
        print(f"  ⚠️  Backend ยังไม่ online — ระบบยังทำงานได้ แต่จะไม่บันทึกผล")
 
    # ─── 4. สรุปคำแนะนำวันพรีเซนต์ ───────────────────────────
    print("\n" + "─"*60)
    print("  📋 Checklist วันพรีเซนต์:")
    print("─"*60)
    checklist = [
        "ใช้ Hotspot จากมือถือเครื่องที่ 2 (ไม่ใช้ Wi-Fi สาธารณะ)",
        "เปิด IP Webcam บนมือถือก่อน → กด Start server",
        "ตั้ง Resolution ที่ 640x480 ใน IP Webcam (Video preferences)",
        "ตั้ง Quality ที่ 50-70% เพื่อลด lag",
        "แล็ปท็อปและมือถือห่างกันไม่เกิน 5 เมตร",
        "ปิด App อื่น ๆ บนแล็ปท็อปที่กิน CPU",
        "รัน network_check.py ก่อนพรีเซนต์ 10 นาที",
        "รัน stream_main.py และกด q เพื่อทดสอบระบบ",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"  {'✓' if i <= 4 else '○'} {i}. {item}")
 
    print("\n" + "─"*60)
    print("  IP Webcam App Settings (แนะนำ):")
    print("─"*60)
    settings = {
        "Resolution":    "640x480",
        "Quality":       "50-70%",
        "FPS limit":     "15-20 fps  (ประหยัด bandwidth)",
        "Video encoder": "MJPEG  (เสถียรกว่า H.264 ผ่าน HTTP)",
        "Port":          "8080  (default)",
        "Audio":         "ปิด  (ไม่จำเป็น ลด bandwidth)",
    }
    for k, v in settings.items():
        print(f"    {k:<18}: {v}")
 
    print("\n" + "═"*60 + "\n")
 
 
if __name__ == "__main__":
    main()