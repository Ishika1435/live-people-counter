# Live People Counter – Real-Time Head Detection & Tracking

A real-time **head-based people counting system** using **YOLOv5** for detection and **DeepSORT / IoU tracking** for stable ID assignment.  
Designed for CCTV and live monitoring to measure **how many people are present** in the camera frame at any moment.

This project provides two versions of the tracking pipeline:

- **DeepSORT (main branch)** → High stability, ideal for dense crowds  
- **IoU Tracker (iou-version branch)** → Lightweight and fast for simple scenes  

---

## ⭐ Features
- Real-time head detection using YOLOv5  
- Continuous count of people present in the frame  
- Multi-object tracking  
- Two tracking modes:
  - **DeepSORT** → stable IDs, handles occlusion  
  - **IoU Tracking** → simple & lightweight  
- Basic HTML templates (`index.html` and `report.html`) included  
- Easy to extend for entry/exit counting  

> **Note:** Line-crossing (entry count) is **only implemented yet**.

---

## 🧠 Why Two Versions?

### **1. DeepSORT (main)**
Suitable for:
- High-density environments  
- People crossing/occluding  
- Cases requiring stable ID tracking  

Uses:
- Kalman Filter (motion prediction)  
- IoU matching  
- Re-identification embeddings (appearance features)  

---

### **2. IoU Tracker (iou-version)**
Suitable for:
- Low/medium crowd  
- Faster execution  
- Simpler systems  

Uses only:
- Bounding-box IoU overlap  
- No appearance features  
- Extremely fast but less stable in crowded scenes  

---


## 📁 Repository Structure

- `app.py` — DeepSORT tracking (ENTRY implemented)     
- `templates/index.html` — Main UI  
- `templates/report.html` — Report UI  
- `output/` — Saved outputs  


---

### **IoU Version (`iou-version`)**

- `iou.py` — IoU tracking (ENTRY implemented)   
- `templates/index.html` — Main UI  
- `templates/report.html` — Report UI  
- `output/` — Saved outputs 


---

## 🚀 How to Run (DeepSORT Version)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2. Add YOLOv5 model
```bash
Place your trained `.pt` file here (not uploaded to GitHub):
```

Ensure the path is correctly set inside `app.py`.

### 3. Run the application
```bash
python app.py
```

---

## 🚀 How to Run (IoU Version)

### 1. Switch to IoU branch
```bash
git checkout iou-version
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add YOLOv5 model
```bash
Place your trained `.pt` file here (not uploaded to GitHub):
```
Ensure the path is correctly set inside `iou.py`.


### 4. Start IoU tracker
```bash
python iou.py
```

---

## 🔮 Future Enhancements

- **Exit counting**
- ByteTrack integration  
- Multi-camera analytics  
- Real-time dashboards  
- Database logging  
- Cloud API endpoints  

---

## 🧪 Tech Stack

- Python  
- YOLOv5  
- OpenCV  
- DeepSORT 
- IoU Tracker  
- Flask  
