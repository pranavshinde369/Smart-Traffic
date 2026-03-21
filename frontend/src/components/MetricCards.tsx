import { useEffect, useRef, useState } from 'react';
import { Car, TrafficCone, Leaf, type LucideIcon } from 'lucide-react';

interface MetricCardProps {
  icon: LucideIcon;
  title: string;
  value: string;
  subtitle: string;
  accentClass: string;
  progress?: number;
  isOffline?: boolean;
}

function MetricCard({ icon: Icon, title, value, subtitle, accentClass, progress, isOffline }: MetricCardProps) {
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
          <span className={`text-[44px] leading-none font-bold font-mono text-foreground ${animate ? 'metric-count-up' : ''}`}>
            {value}
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

function SignalPhaseTimer({ dynamicGreenTime, isOffline }: { dynamicGreenTime: number; isOffline: boolean }) {
  const [countdown, setCountdown] = useState(dynamicGreenTime);
  const totalRef = useRef(dynamicGreenTime);

  useEffect(() => {
    if (dynamicGreenTime > 0) {
      totalRef.current = dynamicGreenTime;
      setCountdown(dynamicGreenTime);
    }
  }, [dynamicGreenTime]);

  useEffect(() => {
    if (isOffline || totalRef.current <= 0) return;
    const t = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) return totalRef.current;
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(t);
  }, [isOffline]);

  const total = totalRef.current || 1;
  const fraction = countdown / total;
  const ringColor =
    fraction > 0.3 ? 'hsl(var(--success))' :
    fraction > 0.1 ? 'hsl(var(--warning))' :
    'hsl(var(--alert))';

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - fraction);

  return (
    <div className="dashboard-card flex flex-col items-center gap-3">
      <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider self-start">
        Current Signal Phase
      </span>
      <div className="relative w-[140px] h-[140px]">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r={radius} fill="none" stroke="hsl(var(--secondary))" strokeWidth="8" />
          <circle
            cx="60" cy="60" r={radius} fill="none"
            stroke={isOffline ? 'hsl(var(--muted))' : ringColor}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={isOffline ? circumference : offset}
            className="transition-all duration-1000 ease-linear"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {isOffline ? (
            <span className="text-2xl font-bold font-mono text-alert">--</span>
          ) : (
            <>
              <span className="text-[36px] leading-none font-bold font-mono text-foreground">{countdown}</span>
              <span className="text-xs text-muted-foreground mt-1">sec</span>
            </>
          )}
        </div>
      </div>
      <span className="text-xs text-muted-foreground text-center">Market Yard Junction — Active Phase</span>
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

export function MetricCards({ vehicleCount, dynamicGreenTime, carbonSavedKg, error }: MetricCardsProps) {
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
      <SignalPhaseTimer dynamicGreenTime={dynamicGreenTime} isOffline={offline} />
    </div>
  );
}
