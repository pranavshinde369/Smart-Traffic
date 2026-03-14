import { useState } from 'react';
import { Flame, MoonStar, Zap } from 'lucide-react';

export function DashboardHeader() {
  const [yatraMode, setYatraMode] = useState(false);

  return (
    <>
      <header className="sticky top-0 z-40 bg-card/80 backdrop-blur-md border-b border-border px-6 py-3 flex items-center justify-between">
        {/* Breadcrumb */}
        <span className="text-sm text-muted-foreground">
          Live Monitoring / <span className="text-foreground">Market Yard Junction</span>
        </span>

        {/* AI Status */}
        <div className="flex items-center gap-2 text-sm">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-success" />
          </span>
          <span className="text-muted-foreground">AI Engine <span className="text-success font-medium">Active</span></span>
        </div>

        {/* Yatra Toggle */}
        <button
          onClick={() => setYatraMode(!yatraMode)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            yatraMode
              ? 'bg-warning/20 text-warning border border-warning/40 yatra-glow'
              : 'bg-secondary text-muted-foreground border border-border hover:text-foreground'
          }`}
        >
          {yatraMode ? <Flame className="w-4 h-4" /> : <MoonStar className="w-4 h-4" />}
          {yatraMode ? 'YATRA MODE ACTIVE' : 'Yatra Mode: OFF'}
        </button>
      </header>

      {/* Yatra Banner */}
      {yatraMode && (
        <div className="bg-warning/10 border-b border-warning/30 px-6 py-3 flex items-center gap-2 text-warning text-sm yatra-glow">
          <Zap className="w-4 h-4 flex-shrink-0" />
          <span className="font-medium">
            YATRA PROTOCOL ACTIVE — Siddheshwar Yatra routes have been cleared. All signals on procession corridors set to GREEN WAVE.
          </span>
        </div>
      )}
    </>
  );
}
