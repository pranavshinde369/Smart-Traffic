import { useState, useEffect } from 'react';
import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { Wifi, Loader2, Camera } from 'lucide-react';

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

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    setError(false);
  }, [selectedCamera]);

  return (
    <div className="flex min-h-screen bg-background">
      <DashboardSidebar />
      <main className="flex-1 ml-[220px] transition-all duration-300 min-h-screen flex flex-col">
        <DashboardHeader />
        <div className="flex-1 p-6 flex flex-col gap-4">
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

          {/* Fullscreen feed */}
          <div className="flex-1 dashboard-card relative overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-foreground">
                Live AI CCTV Feed — {cameras.find(c => c.id === selectedCamera)?.label}
              </h3>
              <span className="text-xs bg-success/20 text-success px-2 py-0.5 rounded-full font-mono">
                YOLOv8n Active
              </span>
            </div>

            <div className="relative flex-1 min-h-[400px] rounded-lg overflow-hidden bg-secondary">
              {error ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
                  <Wifi className="w-10 h-10" />
                  <span className="text-sm">Connecting to AI Engine...</span>
                  <Loader2 className="w-5 h-5 animate-spin" />
                </div>
              ) : (
                <img
                  src={`http://localhost:8000/api/video-feed`}
                  alt="Live CCTV Feed"
                  className="w-full h-full object-cover"
                  onError={() => setError(true)}
                />
              )}

              <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-background/70 backdrop-blur-sm px-2 py-1 rounded text-xs">
                <span className="w-2 h-2 rounded-full bg-alert pulse-live" />
                <span className="text-alert font-bold font-mono">LIVE</span>
              </div>

              <div className="absolute bottom-3 right-3 bg-background/70 backdrop-blur-sm px-2 py-1 rounded text-xs text-muted-foreground font-mono">
                {time.toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default CameraFeed;
