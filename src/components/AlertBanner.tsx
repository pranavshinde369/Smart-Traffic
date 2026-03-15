import { useState, useEffect } from 'react';
import { X } from 'lucide-react';

interface AlertBannerProps {
  encroachmentAlert: boolean;
  error: string | null;
}

export function AlertBanner({ encroachmentAlert, error }: AlertBannerProps) {
  const [dismissed, setDismissed] = useState(false);
  const [challanId, setChallanId] = useState('');

  // Reset dismissed state when alert status changes
  useEffect(() => {
    if (encroachmentAlert) {
      setDismissed(false);
      setChallanId(`SMC2026-${Date.now()}`);
    }
  }, [encroachmentAlert]);

  if (error) {
    return (
      <div className="dashboard-card border-warning/40 bg-warning/5 flex items-center gap-3 text-sm text-warning">
        <span>⚠️</span>
        <span className="font-medium">Backend Offline — Start the Python server at localhost:8000</span>
      </div>
    );
  }

  if (encroachmentAlert && !dismissed) {
    return (
      <div className="dashboard-card border-alert/40 bg-alert/5 alert-border-pulse flex items-center gap-3 text-sm text-alert animate-fade-in">
        <span className="text-lg flex-shrink-0">🚨</span>
        <span className="font-medium flex-1">
          ENCROACHMENT DETECTED — No-Parking Zone Breach at Market Yard Junction — E-Challan #{challanId} Auto-Drafted
        </span>
        <span className="text-xs font-mono text-alert/70 flex-shrink-0">
          {new Date().toLocaleTimeString()}
        </span>
        <button
          onClick={() => setDismissed(true)}
          className="flex-shrink-0 w-7 h-7 rounded-md bg-alert/10 hover:bg-alert/20 flex items-center justify-center transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="dashboard-card border-success/30 bg-success/5 flex items-center gap-3 text-sm text-success animate-fade-in">
      <span>✅</span>
      <span className="font-medium">All Clear — No violations detected</span>
    </div>
  );
}
