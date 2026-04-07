import { AlertTriangle, ShieldCheck, TrendingUp } from 'lucide-react';

interface PredictionBannerProps {
  prediction: string;
}

export function PredictionBanner({ prediction }: PredictionBannerProps) {
  if (!prediction) return null;

  const upper = prediction.toUpperCase();
  const isHigh = upper.includes('HIGH');
  const isModerate = upper.includes('MODERATE');

  const config = isHigh
    ? { icon: AlertTriangle, classes: 'border-destructive/40 bg-destructive/10 text-destructive', badge: 'bg-destructive/20 text-destructive flash-alert', dot: 'bg-destructive' }
    : isModerate
    ? { icon: TrendingUp, classes: 'border-warning/40 bg-warning/10 text-warning', badge: 'bg-warning/20 text-warning', dot: 'bg-warning' }
    : { icon: ShieldCheck, classes: 'border-success/40 bg-success/10 text-success', badge: 'bg-success/20 text-success', dot: 'bg-success' };

  const Icon = config.icon;

  return (
    <div className={`dashboard-card flex items-center gap-3 ${config.classes}`}>
      <div className={`h-8 w-8 rounded-lg flex items-center justify-center ${config.badge}`}>
        <Icon className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
          AI Congestion Forecast — Next 15 Minutes
        </p>
        <p className="text-sm font-semibold truncate">{prediction}</p>
      </div>
      <span className={`relative flex h-2.5 w-2.5 ${isHigh ? '' : ''}`}>
        {isHigh && <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: 'hsl(var(--destructive))' }} />}
        <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${config.dot}`} />
      </span>
    </div>
  );
}
