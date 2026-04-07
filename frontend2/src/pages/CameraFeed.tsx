import { useState, useEffect, useCallback } from 'react';
import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { Wifi, Loader2, Camera, Info } from 'lucide-react';

const cameras = [
  { id: '04', label: 'Camera 04 — Market Yard Junction' },
  { id: '01', label: 'Camera 01 — Navi Peth Crossing' },
  { id: '02', label: 'Camera 02 — Station Road' },
  { id: '03', label: 'Camera 03 — Hutatma Chowk' },
];

const CameraFeed = () => {
  const [selectedCamera, setSelectedCamera] = useState('04');
  const [error, setError] = useState(false);
  const [time, setTime] = useState(new Date());
  const [fps, setFps] = useState(26);
  const [latency, setLatency] = useState(15);
  const [vehicleCount, setVehicleCount] = useState(0);

  useEffect(() => {
    const t = setInterval(() => {
      setTime(new Date());
      setFps(24 + Math.floor(Math.random() * 5));
      setLatency(12 + Math.floor(Math.random() * 7));
    }, 1000);
    return () => clearInterval(t);
  }, []);

  const fetchVehicles = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/api/traffic-data');
      if (!res.ok) return;
      const json = await res.json();
      setVehicleCount(json.vehicle_count ?? 0);
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    fetchVehicles();
    const t = setInterval(fetchVehicles, 1000);
    return () => clearInterval(t);
  }, [fetchVehicles]);

  useEffect(() => {
    setError(false);
  }, [selectedCamera]);

  const statItems = [
    { label: 'FPS', value: `${fps}` },
    { label: 'Resolution', value: '640×480' },
    { label: 'Detections', value: String(vehicleCount) },
    { label: 'Latency', value: `${latency}ms` },
  ];

  return (
    <div className="flex min-h-screen bg-background">
      <DashboardSidebar />
      <main className="flex-1 ml-[220px] transition-all duration-300 min-h-screen flex flex-col">
        <DashboardHeader />
        <div className="flex-1 p-6 flex flex-col gap-4 overflow-y-auto">
          {/* Camera selector */}
          <div className="flex items-center gap-3 flex-wrap">
            {cameras.map((cam) => (
              <button
                key={cam.id}
                onClick={() => setSelectedCamera(cam.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                  selectedCamera === cam.id
                    ? 'bg-primary/15 text-primary border-primary/40'
                    : 'bg-card text-muted-foreground border-border hover:text-foreground hover:border-muted-foreground/30'
                }`}
              >
                <Camera className="w-4 h-4" />
                {cam.label}
              </button>
            ))}
          </div>

          {/* Stats bar */}
          <div className="flex items-center gap-6 px-4 py-2.5 bg-card rounded-lg border border-border">
            {statItems.map((s) => (
              <div key={s.label} className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{s.label}</span>
                <span className="text-sm font-mono font-semibold text-foreground">{s.value}</span>
              </div>
            ))}
          </div>

          {/* Video feed */}
          <div className="dashboard-card relative overflow-hidden">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-foreground">
                Live AI CCTV Feed — {cameras.find(c => c.id === selectedCamera)?.label}
              </h3>
              <span className="text-xs bg-success/20 text-success px-2 py-0.5 rounded-full font-mono">
                YOLOv8n Active
              </span>
            </div>

            <div className="relative w-full rounded-lg overflow-hidden bg-secondary" style={{ aspectRatio: '16/9', maxHeight: 480 }}>
              {error ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
                  <Wifi className="w-10 h-10" />
                  <span className="text-sm">Connecting to AI Engine...</span>
                  <Loader2 className="w-5 h-5 animate-spin" />
                </div>
              ) : (
                <img
                  src="http://localhost:8000/api/video-feed"
                  alt="Live CCTV Feed"
                  className="w-full h-full object-cover"
                  onError={() => setError(true)}
                />
              )}

              {/* Scanline overlay */}
              <div className="absolute inset-0 pointer-events-none scanline-overlay" />

              <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-background/70 backdrop-blur-sm px-2 py-1 rounded text-xs">
                <span className="w-2 h-2 rounded-full bg-alert pulse-live" />
                <span className="text-alert font-bold font-mono">LIVE</span>
              </div>

              <div className="absolute bottom-3 right-3 bg-background/70 backdrop-blur-sm px-2 py-1 rounded text-xs text-muted-foreground font-mono">
                {time.toLocaleTimeString()}
              </div>
            </div>
          </div>

          {/* Info card */}
          <div className="flex items-start gap-3 px-4 py-3 bg-card rounded-lg border border-border text-xs text-muted-foreground">
            <Info className="w-4 h-4 flex-shrink-0 mt-0.5 text-primary" />
            <span>
              All cameras share the same AI engine backend. In production, each camera runs an independent YOLO inference thread.
            </span>
          </div>
        </div>
      </main>
    </div>
  );
};

export default CameraFeed;
