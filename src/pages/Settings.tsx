import { useState } from 'react';
import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Save, RotateCcw } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

const defaultZones = [
  { name: 'Market Yard Junction', coords: '18.5196, 75.9382' },
  { name: 'Navi Peth', coords: '18.5120, 75.9250' },
  { name: 'Station Road', coords: '18.5098, 75.9310' },
];

const Settings = () => {
  const { toast } = useToast();
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const [confidence, setConfidence] = useState([0.45]);
  const [zones, setZones] = useState(defaultZones.map(z => ({ ...z })));

  const handleSave = () => {
    toast({ title: 'Settings saved', description: 'Configuration updated successfully.' });
  };

  const handleReset = () => {
    setBackendUrl('http://localhost:8000');
    setConfidence([0.45]);
    setZones(defaultZones.map(z => ({ ...z })));
    toast({ title: 'Settings reset', description: 'All values restored to defaults.' });
  };

  return (
    <div className="flex min-h-screen bg-background">
      <DashboardSidebar />
      <main className="flex-1 ml-[220px] transition-all duration-300 min-h-screen flex flex-col">
        <DashboardHeader />
        <div className="flex-1 p-6 space-y-6 overflow-y-auto max-w-3xl">
          <h2 className="text-lg font-semibold text-foreground">System Settings</h2>

          {/* Backend URL */}
          <div className="dashboard-card space-y-4">
            <h3 className="text-sm font-semibold text-foreground">Backend Configuration</h3>
            <div className="space-y-2">
              <Label htmlFor="backend-url">Backend API URL</Label>
              <Input
                id="backend-url"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                placeholder="http://localhost:8000"
              />
              <p className="text-xs text-muted-foreground">
                Base URL for the Python AI backend server.
              </p>
            </div>
          </div>

          {/* Confidence threshold */}
          <div className="dashboard-card space-y-4">
            <h3 className="text-sm font-semibold text-foreground">Detection Settings</h3>
            <div className="space-y-3">
              <Label>YOLOv8 Confidence Threshold</Label>
              <div className="flex items-center gap-4">
                <Slider
                  value={confidence}
                  onValueChange={setConfidence}
                  min={0.1}
                  max={0.95}
                  step={0.05}
                  className="flex-1"
                />
                <span className="text-sm font-mono text-primary w-12 text-right">
                  {confidence[0].toFixed(2)}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Higher values reduce false positives. Recommended: 0.40–0.60.
              </p>
            </div>
          </div>

          {/* Zone coordinates */}
          <div className="dashboard-card space-y-4">
            <h3 className="text-sm font-semibold text-foreground">Zone Coordinates</h3>
            <p className="text-xs text-muted-foreground">
              Define monitoring zone coordinates (latitude, longitude) for encroachment detection.
            </p>
            <div className="space-y-3">
              {zones.map((zone, i) => (
                <div key={i} className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs">Zone Name</Label>
                    <Input
                      value={zone.name}
                      onChange={(e) => {
                        const updated = [...zones];
                        updated[i] = { ...updated[i], name: e.target.value };
                        setZones(updated);
                      }}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Coordinates</Label>
                    <Input
                      value={zone.coords}
                      onChange={(e) => {
                        const updated = [...zones];
                        updated[i] = { ...updated[i], coords: e.target.value };
                        setZones(updated);
                      }}
                      placeholder="lat, lng"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <Button onClick={handleSave} className="gap-2">
              <Save className="w-4 h-4" />
              Save Settings
            </Button>
            <Button variant="outline" onClick={handleReset} className="gap-2">
              <RotateCcw className="w-4 h-4" />
              Reset Defaults
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Settings;
