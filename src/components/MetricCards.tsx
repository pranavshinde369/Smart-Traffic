import { useEffect, useRef, useState } from 'react';
import { Car, TrafficCone, Leaf, Activity, type LucideIcon } from 'lucide-react';

interface MetricCardProps {
  icon: LucideIcon;
  title: string;
  value: string;
  subtitle: string;
  accentClass: string;
  progress?: number;
  statusDot?: 'online' | 'initializing' | null;
  isOffline?: boolean;
}

function MetricCard({ icon: Icon, title, value, subtitle, accentClass, progress, statusDot, isOffline }: MetricCardProps) {
  const [animate, setAnimate] = useState(false);
  const prevValue = useRef(value);

  useEffect(() => {
    if (prevValue.current !== value) {
      setAnimate(true);
      prevValue.current = value;
      const t = setTimeout(() => setAnimate(false), 400);
      return () => clearTimeout(t);
    }
  }, [value]);

  return (
    <div className="dashboard-card flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider">{title}</span>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${accentClass}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className="flex items-end gap-2">
        {isOffline ? (
          <span className="text-[44px] leading-none font-bold font-mono text-alert">Offline</span>
        ) : (
          <span
            className={`text-[44px] leading-none font-bold font-mono text-foreground ${animate ? 'metric-count-up' : ''}`}
          >
            {value}
          </span>
        )}
        {!isOffline && statusDot && (
          <span className={`text-xs font-medium flex items-center gap-1 mb-1 ${
            statusDot === 'online' ? 'text-success' : 'text-warning'
          }`}>
            {statusDot === 'online' ? '●' : '◌'} {statusDot === 'online' ? 'ONLINE' : 'INITIALIZING'}
          </span>
        )}
      </div>
      <span className="text-xs text-muted-foreground">{subtitle}</span>
      {progress !== undefined && !isOffline && (
        <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden">
          <div
            className="h-full bg-success rounded-full transition-all duration-500"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

interface MetricCardsProps {
  vehicleCount: number;
  dynamicGreenTime: number;
  carbonSavedKg: number;
  backendStatus: string;
  error: string | null;
}

export function MetricCards({ vehicleCount, dynamicGreenTime, carbonSavedKg, backendStatus, error }: MetricCardsProps) {
  const offline = !!error;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard
        icon={Car}
        title="Live Traffic Density"
        value={String(vehicleCount)}
        subtitle="Vehicles detected in frame"
        accentClass="bg-primary/20 text-primary"
        isOffline={offline}
      />
      <MetricCard
        icon={TrafficCone}
        title="RL Green Signal Time"
        value={`${dynamicGreenTime}s`}
        subtitle="RL-Optimized duration"
        accentClass="bg-success/20 text-success"
        progress={(dynamicGreenTime / 120) * 100}
        isOffline={offline}
      />
      <MetricCard
        icon={Leaf}
        title="Eco-Niyantran"
        value={`${carbonSavedKg} kg`}
        subtitle="CO₂ emissions saved today"
        accentClass="bg-emerald-500/20 text-emerald-400"
        isOffline={offline}
      />
      <MetricCard
        icon={Activity}
        title="System Status"
        value=""
        subtitle="Backend connection status"
        accentClass="bg-teal-500/20 text-teal-400"
        statusDot={offline ? null : backendStatus === 'live' ? 'online' : 'initializing'}
        isOffline={offline}
      />
    </div>
  );
}
