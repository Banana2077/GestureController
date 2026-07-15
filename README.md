# GestureController

ระบบควบคุม Unity Game ด้วยท่ามือผ่านกล้องเว็บแคม โดยใช้ MediaPipe สำหรับตรวจจับมือ และ YOLO สำหรับจำแนกท่าทาง

---

## ความต้องการของระบบ

- **OS**: Windows 10 / 11 (64-bit)
- **Python**: 3.10 หรือ 3.11 (แนะนำ **3.10**)
- **กล้องเว็บแคม**: USB หรือ Built-in Webcam
- **RAM**: ขั้นต่ำ 4 GB (แนะนำ 8 GB ขึ้นไป)

---

## ขั้นตอนที่ 1 — ติดตั้ง Python

> **สำคัญ**: ต้องติดตั้ง Python **3.10.x** หรือ **3.11.x** เท่านั้น (MediaPipe ยังไม่รองรับ Python 3.12+)

### วิธีที่ 1: ดาวน์โหลดจากเว็บไซต์ทางการ

1. ไปที่ https://www.python.org/downloads/release/python-31011/
2. เลื่อนลงไปที่ **Files** แล้วเลือก **Windows installer (64-bit)**
3. รันไฟล์ `.exe` ที่ดาวน์โหลดมา
4.  **ติ๊กช่อง "Add Python to PATH"** ก่อนกด Install
5. กด **Install Now**

### ตรวจสอบว่าติดตั้งสำเร็จ

เปิด **Command Prompt** หรือ **PowerShell** แล้วพิมพ์:

```bash
python --version
```

ควรได้ผลลัพธ์เป็น `Python 3.10.x` หรือ `Python 3.11.x`

---

## ขั้นตอนที่ 2 — ติดตั้ง Library

### เปิด PowerShell / Command Prompt

กด `Win + R` → พิมพ์ `cmd` → กด Enter

### เปลี่ยน Directory ไปที่โฟลเดอร์โปรเจกต์

```bash
cd path\to\GestureController
```

> ตัวอย่าง: `cd C:\Users\YourName\Desktop\GestureController`

### ติดตั้ง Library ทั้งหมดด้วยคำสั่งเดียว

```bash
pip install opencv-python mediapipe scikit-learn==1.6.1 ultralytics numpy
```

#### หรือติดตั้งทีละตัว (หากเกิด Error)

```bash
pip install opencv-python
pip install mediapipe
pip install scikit-learn==1.6.1
pip install ultralytics
pip install numpy
```

---

## รายละเอียด Library ที่ใช้

| Library | เวอร์ชัน | หน้าที่ |
|---|---|---|
| `opencv-python` | Latest | อ่านภาพจากกล้อง, แสดงผล |
| `mediapipe` | Latest | ตรวจจับมือและ Landmark |
| `scikit-learn` | **1.6.1** | โหลด Model `.pkl` |
| `ultralytics` | Latest | รัน YOLO Model `.pt` |
| `numpy` | Latest | คำนวณทางคณิตศาสตร์ |

> **หมายเหตุ**: `scikit-learn` ต้องเป็นเวอร์ชัน **1.6.1 เท่านั้น** เพื่อให้ `model.pkl` โหลดได้ถูกต้อง

---

## ขั้นตอนที่ 3 — รันโปรแกรม

### ตรวจสอบว่ามีไฟล์ครบ

ในโฟลเดอร์ต้องมีไฟล์เหล่านี้:

```
GestureController/
├── main.py          ✅ ไฟล์หลัก
├── pd_mode.py       ✅ โหมด Finger Tracking
├── best.pt          ✅ YOLO Model (ไฟล์ขนาดใหญ่ ~19 MB)
├── model.pkl        ✅ Scikit-learn Model
└── savevalue.py     (เครื่องมือเก็บ Dataset)
```

### รันโปรแกรมหลัก

```bash
python main.py
```

---

## 🎮 วิธีใช้งาน

| การใช้งาน | คำอธิบาย |
|---|---|
| **มือซ้าย (ซีกซ้ายจอ)** | ควบคุมการเคลื่อนที่ (Joystick) — กำมือค้างแล้วขยับ |
| **มือขวา (ซีกขวาจอ)** | AIM / SHOOT — ใช้นิ้วหัวแม่มือ + นิ้วชี้ |
| **กางมือ 2 ข้างค้าง** | สลับไปโหมด **GESTURE** (จำแนกท่ามือด้วย YOLO) |
| **โหมด PD** | ควบคุมได้จาก Unity (minigame) |

### ท่ามือที่รู้จัก (Gesture Mode)

| ท่ามือ | ชื่อ |
|---|---|
| 🐰 | rabbit |
| 🐕 | dog |
| 🐦 | bird |
| 🐄 | cow |
| 🦌 | deer |

---

## 🔌 การเชื่อมต่อกับ Unity

โปรแกรมจะรอรับ Connection จาก Unity อัตโนมัติ:

| Protocol | Port | หน้าที่ |
|---|---|---|
| TCP | `5005` | รับ-ส่ง Gesture และ Mode |
| UDP | `5006` | ส่งคำสั่งควบคุม (Move/Jump/Shoot) |
| UDP | `5052` | ส่งข้อมูล Finger Curl (PD Mode) |

---

##  แก้ปัญหาที่พบบ่อย

### Python หาไม่เจอ / `python` ใช้ไม่ได้
```bash
# ลองใช้ py แทน python
py --version
py main.py
```
หรือ ติดตั้ง Python ใหม่แล้วติ๊ก **"Add Python to PATH"**

---

### `pip` ใช้ไม่ได้
```bash
python -m pip install opencv-python mediapipe scikit-learn==1.6.1 ultralytics numpy
```

---

### `mediapipe` ติดตั้งไม่ได้บน Python 3.12+
ต้องใช้ Python **3.10 หรือ 3.11** เท่านั้น
ดาวน์โหลด: https://www.python.org/downloads/release/python-31011/

---

### กล้องเปิดไม่ได้
- ตรวจสอบว่ากล้องไม่ถูกแอปอื่นใช้อยู่ (เช่น Teams, Zoom)
- ลองเปลี่ยน `cv2.VideoCapture(0)` เป็น `cv2.VideoCapture(1)` ใน `main.py`

---

### `best.pt` หาไม่เจอ
- ตรวจสอบว่าไฟล์ `best.pt` อยู่ในโฟลเดอร์เดียวกับ `main.py`
- รันโปรแกรมจากใน Directory ของโปรเจกต์เสมอ

---

### ติดตั้ง Library ช้า / Timeout
```bash
pip install --timeout 120 opencv-python mediapipe scikit-learn==1.6.1 ultralytics numpy
```

---

## สำหรับนักพัฒนา — เครื่องมือเพิ่มเติม

| ไฟล์ | หน้าที่ |
|---|---|
| `savevalue.py` | เก็บ Dataset รูปภาพมือสำหรับ Train YOLO |
| `train_colab.ipynb` | Train YOLO Model บน Google Colab |
| `prepare_dataset.py` | เตรียม Dataset ก่อน Train |

---

*หากพบปัญหาเพิ่มเติม ติดต่อผู้พัฒนาโปรเจกต์*
