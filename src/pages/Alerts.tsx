import { useState, useEffect, useRef, useCallback } from 'react';
import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { ShieldAlert, ShieldCheck, Clock, FileText, Eye, Car, Bike } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';

interface AlertEntry {
  id: number;
  timestamp: string;
  zone: string;
  status: 'violation' | 'clear';
  message: string;
  challanNo: string;
  vehicleType: 'car' | 'bike';
  bbox: { x: number; y: number; w: number; h: number };
}

const zones = ['Market Yard Junction', 'Navi Peth', 'Chatti Galli', 'Station Road', 'Hutatma Chowk'];

let globalChallanIndex = 40;

function generateChallan(): string {
  globalChallanIndex++;
  const d = new Date();
  const dateStr = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
  return `SMC-${dateStr}-${String(globalChallanIndex).padStart(4, '0')}`;
}

function randomBbox() {
  return {
    x: Math.floor(Math.random() * 400 + 50),
    y: Math.floor(Math.random() * 300 + 50),
    w: Math.floor(Math.random() * 80 + 60),
    h: Math.floor(Math.random() * 60 + 40),
  };
}

function randomVehicle(): 'car' | 'bike' {
  return Math.random() > 0.4 ? 'car' : 'bike';
}

const Alerts = () => {
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<AlertEntry | null>(null);
  const prevEncroachment = useRef<boolean | null>(null);
  const idRef = useRef(0);

  // Seed initial alerts
  useEffect(() => {
    const initial: AlertEntry[] = [
      { id: ++idRef.current, timestamp: new Date(Date.now() - 300000).toLocaleString(), zone: 'Market Yard Junction', status: 'violation', message: 'Encroachment detected in No-Parking Zone', challanNo: generateChallan(), vehicleType: randomVehicle(), bbox: randomBbox() },
      { id: ++idRef.current, timestamp: new Date(Date.now() - 240000).toLocaleString(), zone: 'Market Yard Junction', status: 'clear', message: 'Zone cleared — no violations', challanNo: '—', vehicleType: 'car', bbox: randomBbox() },
      { id: ++idRef.current, timestamp: new Date(Date.now() - 180000).toLocaleString(), zone: 'Navi Peth', status: 'violation', message: 'Encroachment detected in No-Parking Zone', challanNo: generateChallan(), vehicleType: randomVehicle(), bbox: randomBbox() },
      { id: ++idRef.current, timestamp: new Date(Date.now() - 60000).toLocaleString(), zone: 'Station Road', status: 'violation', message: 'Encroachment detected in No-Parking Zone', challanNo: generateChallan(), vehicleType: randomVehicle(), bbox: randomBbox() },
      { id: ++idRef.current, timestamp: new Date().toLocaleString(), zone: 'Station Road', status: 'clear', message: 'Zone cleared — no violations', challanNo: '—', vehicleType: 'car', bbox: randomBbox() },
    ];
    setAlerts(initial);
  }, []);

  // Poll API every 2s
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8000/api/traffic-data');
        if (!res.ok) return;
        const json = await res.json();
        const isEncroachment = json.encroachment_alert === true;

        if (isEncroachment && prevEncroachment.current === false) {
          const zone = zones[Math.floor(Math.random() * zones.length)];
          const vStatus: 'violation' = 'violation';
          setAlerts(prev => [{
            id: ++idRef.current,
            timestamp: new Date().toLocaleString(),
            zone,
            status: vStatus,
            message: 'Encroachment detected in No-Parking Zone',
            challanNo: generateChallan(),
            vehicleType: randomVehicle(),
            bbox: randomBbox(),
          }, ...prev].slice(0, 50));
        }
        prevEncroachment.current = isEncroachment;
      } catch {
        // backend offline — ignore
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Also simulate new alerts periodically
  useEffect(() => {
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
        challanNo: isViolation ? generateChallan() : '—',
        vehicleType: randomVehicle(),
        bbox: randomBbox(),
      }, ...prev].slice(0, 50));
    }, 8000);
    return () => clearInterval(interval);
  }, []);

  const violationCount = alerts.filter(a => a.status === 'violation').length;

  return (
    <div className="flex min-h-screen bg-background">
      <DashboardSidebar />
      <main className="flex-1 ml-[220px] transition-all duration-300 min-h-screen flex flex-col">
        <DashboardHeader />
        <div className="flex-1 p-6 space-y-4 overflow-y-auto">
          <h2 className="text-lg font-semibold text-foreground">Encroachment Alert Log</h2>

          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-4">
            <Card className="bg-card border-border">
              <CardContent className="p-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-alert/10 flex items-center justify-center">
                  <ShieldAlert className="w-5 h-5 text-alert" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Total Violations Today</p>
                  <p className="text-2xl font-bold text-foreground">{violationCount}</p>
                </div>
              </CardContent>
            </Card>
            <Card className="bg-card border-border">
              <CardContent className="p-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">E-Challans Drafted</p>
                  <p className="text-2xl font-bold text-foreground">{violationCount}</p>
                </div>
              </CardContent>
            </Card>
            <Card className="bg-card border-border">
              <CardContent className="p-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center">
                  <ShieldCheck className="w-5 h-5 text-success" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Zones Active</p>
                  <p className="text-2xl font-bold text-foreground">3</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Alert Table */}
          <div className="dashboard-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground text-left">
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">E-Challan</th>
                  <th className="px-4 py-3 font-medium">Timestamp</th>
                  <th className="px-4 py-3 font-medium">Zone</th>
                  <th className="px-4 py-3 font-medium">Details</th>
                  <th className="px-4 py-3 font-medium">Action</th>
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
                    <td className="px-4 py-3 font-mono text-xs text-foreground">
                      {alert.challanNo}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground font-mono text-xs">
                      <span className="flex items-center gap-1.5">
                        <Clock className="w-3 h-3" />
                        {alert.timestamp}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-foreground">{alert.zone}</td>
                    <td className="px-4 py-3 text-muted-foreground">{alert.message}</td>
                    <td className="px-4 py-3">
                      {alert.status === 'violation' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-xs gap-1 text-primary hover:text-primary"
                          onClick={() => setSelectedAlert(alert)}
                        >
                          <Eye className="w-3.5 h-3.5" />
                          View Challan
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Challan Modal */}
      <Dialog open={!!selectedAlert} onOpenChange={(open) => !open && setSelectedAlert(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="font-mono text-base">E-Challan {selectedAlert?.challanNo}</DialogTitle>
            <DialogDescription>Auto-generated violation record</DialogDescription>
          </DialogHeader>
          {selectedAlert && (
            <div className="space-y-4">
              {/* Photo Placeholder */}
              <div className="relative w-full aspect-video bg-muted rounded-lg overflow-hidden flex items-center justify-center border border-border">
                <span className="text-muted-foreground text-sm">Violation Snapshot Placeholder</span>
                {/* Bounding Box Overlay */}
                <div
                  className="absolute border-2 border-alert rounded"
                  style={{
                    left: `${(selectedAlert.bbox.x / 640) * 100}%`,
                    top: `${(selectedAlert.bbox.y / 480) * 100}%`,
                    width: `${(selectedAlert.bbox.w / 640) * 100}%`,
                    height: `${(selectedAlert.bbox.h / 480) * 100}%`,
                  }}
                >
                  <span className="absolute -top-5 left-0 text-[10px] font-mono bg-alert text-alert-foreground px-1 rounded">
                    {selectedAlert.vehicleType === 'car' ? 'CAR' : 'BIKE'}
                  </span>
                </div>
              </div>

              {/* Details Grid */}
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-muted-foreground text-xs">Zone</p>
                  <p className="font-medium text-foreground">{selectedAlert.zone}</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">Timestamp</p>
                  <p className="font-medium text-foreground">{selectedAlert.timestamp}</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">Vehicle Type</p>
                  <p className="font-medium text-foreground flex items-center gap-1.5">
                    {selectedAlert.vehicleType === 'car' ? <Car className="w-4 h-4" /> : <Bike className="w-4 h-4" />}
                    {selectedAlert.vehicleType === 'car' ? 'Car' : 'Bike'}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs">Fine Amount</p>
                  <p className="font-medium text-foreground text-lg">
                    {selectedAlert.vehicleType === 'car' ? '₹500' : '₹300'}
                  </p>
                </div>
                <div className="col-span-2">
                  <p className="text-muted-foreground text-xs">Bounding Box (px)</p>
                  <p className="font-mono text-xs text-foreground">
                    x={selectedAlert.bbox.x} y={selectedAlert.bbox.y} w={selectedAlert.bbox.w} h={selectedAlert.bbox.h}
                  </p>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Alerts;
