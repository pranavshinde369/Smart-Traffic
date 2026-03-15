import { useState, useEffect } from 'react';
import { Wifi, Loader2 } from 'lucide-react';

export function VideoFeed() {
  const [error, setError] = useState(false);
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="dashboard-card relative overflow-hidden">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground">Live AI CCTV Feed — Camera 04</h3>
        <span className="text-xs bg-success/20 text-success px-2 py-0.5 rounded-full font-mono">
          YOLOv8n Active
        </span>
      </div>

      <div className="relative h-[360px] aspect-video rounded-lg overflow-hidden bg-secondary">
        {error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
            <Wifi className="w-8 h-8" />
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

        {/* LIVE badge */}
        <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-background/70 backdrop-blur-sm px-2 py-1 rounded text-xs">
          <span className="w-2 h-2 rounded-full bg-alert pulse-live" />
          <span className="text-alert font-bold font-mono">LIVE</span>
        </div>

        {/* Timestamp */}
        <div className="absolute bottom-3 right-3 bg-background/70 backdrop-blur-sm px-2 py-1 rounded text-xs text-muted-foreground font-mono">
          {time.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}
