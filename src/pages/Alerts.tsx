import { useState, useEffect, useRef } from 'react';
import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { ShieldAlert, ShieldCheck, Clock } from 'lucide-react';

interface AlertEntry {
  id: number;
  timestamp: string;
  zone: string;
  status: 'violation' | 'clear';
  message: string;
}

const zones = ['Market Yard Junction', 'Navi Peth', 'Chatti Galli', 'Station Road', 'Hutatma Chowk'];

const Alerts = () => {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const idRef = useRef(0);

  useEffect(() => {
    // Seed some initial alerts
    const initial: AlertEntry[] = [
      { id: ++idRef.current, timestamp: new Date(Date.now() - 300000).toLocaleString(), zone: 'Market Yard Junction', status: 'violation', message: 'Encroachment detected in No-Parking Zone' },
      { id: ++idRef.current, timestamp: new Date(Date.now() - 240000).toLocaleString(), zone: 'Market Yard Junction', status: 'clear', message: 'Zone cleared — no violations' },
      { id: ++idRef.current, timestamp: new Date(Date.now() - 180000).toLocaleString(), zone: 'Navi Peth', status: 'violation', message: 'Encroachment detected in No-Parking Zone' },
      { id: ++idRef.current, timestamp: new Date(Date.now() - 60000).toLocaleString(), zone: 'Station Road', status: 'violation', message: 'Encroachment detected in No-Parking Zone' },
      { id: ++idRef.current, timestamp: new Date().toLocaleString(), zone: 'Station Road', status: 'clear', message: 'Zone cleared — no violations' },
    ];
    setAlerts(initial);

    // Simulate new alerts coming in
    const interval = setInterval(() => {
      const zone = zones[Math.floor(Math.random() * zones.length)];
      const isViolation = Math.random() > 0.5;
      const status: 'violation' | 'clear' = isViolation ? 'violation' : 'clear';
      setAlerts(prev => [{
        id: ++idRef.current,
        timestamp: new Date().toLocaleString(),
        zone,
        status,
        message: isViolation ? 'Encroachment detected in No-Parking Zone' : 'Zone cleared — no violations',
      }, ...prev].slice(0, 50));
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex min-h-screen bg-background">
      <DashboardSidebar />
      <main className="flex-1 ml-[220px] transition-all duration-300 min-h-screen flex flex-col">
        <DashboardHeader />
        <div className="flex-1 p-6 space-y-4 overflow-y-auto">
          <h2 className="text-lg font-semibold text-foreground">Encroachment Alert Log</h2>

          <div className="dashboard-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground text-left">
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Timestamp</th>
                  <th className="px-4 py-3 font-medium">Zone</th>
                  <th className="px-4 py-3 font-medium">Details</th>
                </tr>
              </thead>
              <tbody>
                {alerts.map((alert) => (
                  <tr key={alert.id} className="border-b border-border/50 transition-colors hover:bg-accent/30">
                    <td className="px-4 py-3">
                      {alert.status === 'violation' ? (
                        <span className="flex items-center gap-1.5 text-alert">
                          <ShieldAlert className="w-4 h-4" />
                          <span className="text-xs font-medium">VIOLATION</span>
                        </span>
                      ) : (
                        <span className="flex items-center gap-1.5 text-success">
                          <ShieldCheck className="w-4 h-4" />
                          <span className="text-xs font-medium">CLEAR</span>
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground font-mono text-xs">
                      <span className="flex items-center gap-1.5">
                        <Clock className="w-3 h-3" />
                        {alert.timestamp}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-foreground">{alert.zone}</td>
                    <td className="px-4 py-3 text-muted-foreground">{alert.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Alerts;
