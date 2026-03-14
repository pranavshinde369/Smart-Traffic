interface AlertBannerProps {
  encroachmentAlert: boolean;
  error: string | null;
}

export function AlertBanner({ encroachmentAlert, error }: AlertBannerProps) {
  if (error) {
    return (
      <div className="dashboard-card border-warning/40 bg-warning/5 flex items-center gap-3 text-sm text-warning">
        <span>⚠️</span>
        <span className="font-medium">Backend Offline — Start the Python server at localhost:8000</span>
      </div>
    );
  }

  if (encroachmentAlert) {
    return (
      <div className="dashboard-card border-alert/40 bg-alert/5 flash-alert glow-red flex items-center gap-3 text-sm text-alert">
        <span>🚨</span>
        <span className="font-medium">
          ENCROACHMENT DETECTED in No-Parking Zone — E-Challan Drafted Automatically
        </span>
        <span className="ml-auto text-xs font-mono text-alert/70">
          {new Date().toLocaleTimeString()}
        </span>
      </div>
    );
  }

  return (
    <div className="dashboard-card border-success/30 bg-success/5 flex items-center gap-3 text-sm text-success">
      <span>✅</span>
      <span className="font-medium">No Violations Detected — Zone Clear</span>
    </div>
  );
}
