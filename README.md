# Smart Traffic 🚦
### AI-powered Smart City Traffic Management Platform

> An intelligent traffic management system powered by YOLOv8 vehicle detection, dynamic signal optimization, and real-time monitoring to reduce congestion and emissions.

---

## 🎯 Overview

Smart Traffic is a comprehensive solution for modern traffic management that combines:
- **Real-time Vehicle Detection** using YOLOv8 computer vision
- **Dynamic Green Signal Optimization** based on live traffic density
- **Virtual Geofencing** with encroachment detection and e-challan alerts
- **Carbon Emission Tracking** to measure environmental impact
- **Live Dashboard** for traffic operators and city planners

---

## 📸 Screenshots

### Dashboard Preview
![Dashboard](./screenshots/dashboard.png)

### Real-time Traffic Stream
![Traffic Feed](./screenshots/traffic-feed.png)

### Traffic Metrics & Alerts
![Metrics](./screenshots/metrics.png)

---

## 🏗️ Project Architecture

```
smart-traffic/
├── backend/
│   ├── main.py                    ← FastAPI + YOLOv8 + OpenCV engine
│   ├── requirements.txt           ← Python dependencies
│   └── traffic_feed.mp4           ← (Add your traffic video here)
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts             ← React + Vite configuration
├── screenshots/                   ← Project screenshots
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Node.js 16+** (for frontend)
- **pip** and **npm**

### Backend Setup

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) Add your traffic video
# Place your video file as: traffic_feed.mp4
# If missing, backend auto-runs with synthetic demo data

# 5. Start the backend server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Backend API**: `http://localhost:8000`

### Frontend Setup

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

**Frontend**: `http://localhost:5173` (or as shown in terminal)

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/video-feed` | GET | MJPEG stream for real-time traffic video |
| `/api/traffic-data` | GET | Live JSON traffic metrics and status |
| `/docs` | GET | Interactive Swagger UI documentation |
| `/health` | GET | Backend health check |

### API Response Example

```json
{
  "vehicle_count": 7,
  "encroachment_alert": true,
  "dynamic_green_time": 21,
  "carbon_saved_kg": 0.0117,
  "timestamp": "2026-04-27T10:30:00Z",
  "backend_status": "live"
}
```

---

## 🧠 Core Features

### 1️⃣ Real-time Vehicle Detection
- **Model**: YOLOv8n (lightweight, ~6MB)
- **Auto-downloads** on first run
- Detects and counts vehicles in real-time
- Processes video in a daemon thread (non-blocking)

### 2️⃣ Virtual Geofencing (Encroachment Shield)
```
Zone Definition: ZONE = (50, 200, 350, 450)
```
- Monitors restricted traffic zones
- Detects vehicles crossing boundaries
- Triggers alerts and generates e-challan drafts
- Red bounding boxes on detected encroachments

### 3️⃣ Dynamic Green Signal Optimization (RL-Optimizer)
```
dynamic_green_time = min(max(vehicle_count × 3, 20s), 120s)
```
- Adapts signal timing based on traffic density
- Minimum: 20 seconds | Maximum: 120 seconds
- Reduces wait times and congestion

### 4️⃣ Eco-Niyantran (Carbon Footprint Tracking)
```
time_saved = max(0, 60 - dynamic_green_time)
carbon_saved = vehicle_count × time_saved × 0.0028 kg/vehicle/sec
```
- Calculates emissions saved vs. fixed 60s signals
- Environmental impact dashboard
- Supports sustainable city goals

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, YOLOv8, OpenCV, Python |
| **Frontend** | React, TypeScript, Vite, CSS |
| **Detection** | YOLOv8n (Ultralytics) |
| **Video Processing** | OpenCV (cv2) |
| **API** | REST API with Swagger UI |

---

## 📊 Data Flow

1. **Input**: Traffic video stream (MP4 or webcam)
2. **Processing**: YOLOv8 detects vehicles in each frame
3. **Analysis**: Calculates metrics (count, geofence violations, signal timing)
4. **Output**: API response + video stream with annotations
5. **Display**: Frontend renders live feed, metrics, and alerts

---

## ⚙️ Configuration

### Video Input
Place your traffic video as `traffic_feed.mp4` in the backend directory. If not provided, a synthetic demo will run automatically.

### Geofence Zone
Modify the `ZONE` variable in `main.py`:
```python
ZONE = (x_min, y_min, x_max, y_max)  # Edit coordinates as needed
```

### Signal Timing
Adjust the dynamic green time calculation:
```python
GREEN_TIME_MIN = 20  # seconds
GREEN_TIME_MAX = 120  # seconds
TIME_PER_VEHICLE = 3  # seconds
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Fork the repository
- Create a feature branch (`git checkout -b feature/amazing-feature`)
- Commit your changes (`git commit -m 'Add amazing feature'`)
- Push to the branch (`git push origin feature/amazing-feature`)
- Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Pranav Shinde** (@pranavshinde369)

---

## 📞 Support & Feedback

For questions, issues, or suggestions, please:
- Open an [Issue](https://github.com/pranavshinde369/Smart-Traffic/issues)
- Check existing [Discussions](https://github.com/pranavshinde369/Smart-Traffic/discussions)
- Contact via GitHub

---

## 🎓 Learn More

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
