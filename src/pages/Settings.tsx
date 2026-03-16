import { useState } from 'react';
import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import { Save, RotateCcw, Info } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ZonePixel {
  name: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

const defaultZones: ZonePixel[] = [
  { name: 'Market Yard Junction', x1: 80, y1: 120, x2: 400, y2: 360 },
  { name: 'Navi Peth', x1: 50, y1: 90, x2: 350, y2: 310 },
  { name: 'Station Road', x1: 100, y1: 100, x2: 450, y2: 380 },
];

const allJunctions = [
  'Market Yard Junction',
  'Navi Peth',
  'Chatti Galli',
  'Station Road',
  'Hutatma Chowk',
];

const Settings = () => {
  const { toast } = useToast();
  const [backendUrl, setBackendUrl] = useState(() => localStorage.getItem('settings_backendUrl') || 'http://localhost:8000');
  const [confidence, setConfidence] = useState<number[]>(() => {
    const saved = localStorage.getItem('settings_confidence');
    return saved ? [parseFloat(saved)] : [0.45];
  });
  const [zones, setZones] = useState<ZonePixel[]>(() => {
    const saved = localStorage.getItem('settings_zones');
    return saved ? JSON.parse(saved) : defaultZones.map(z => ({ ...z }));
  });

  // Yatra Mode state
  const [overrideSignalTime, setOverrideSignalTime] = useState(() => {
    const saved = localStorage.getItem('settings_overrideSignalTime');
    return saved ? parseInt(saved, 10) : 45;
  });
  const [activeJunctions, setActiveJunctions] = useState<string[]>(() => {
    const saved = localStorage.getItem('settings_activeJunctions');
    return saved ? JSON.parse(saved) : ['Market Yard Junction', 'Station Road'];
  });
  const [emergencyContact, setEmergencyContact] = useState(() => localStorage.getItem('settings_emergencyContact') || '');

  const handleSave = () => {
    localStorage.setItem('settings_backendUrl', backendUrl);
    localStorage.setItem('settings_confidence', confidence[0].toString());
    localStorage.setItem('settings_zones', JSON.stringify(zones));
    localStorage.setItem('settings_overrideSignalTime', overrideSignalTime.toString());
    localStorage.setItem('settings_activeJunctions', JSON.stringify(activeJunctions));
    localStorage.setItem('settings_emergencyContact', emergencyContact);
    toast({
      title: '✅ Settings saved',
      description: 'Changes will apply on next backend restart.',
      className: 'border-success/40 bg-success/10 text-success',
    });
  };

  const handleReset = () => {
    setBackendUrl('http://localhost:8000');
    setConfidence([0.45]);
    setZones(defaultZones.map(z => ({ ...z })));
    setOverrideSignalTime(45);
    setActiveJunctions(['Market Yard Junction', 'Station Road']);
    setEmergencyContact('');
    toast({ title: 'Settings reset', description: 'All values restored to defaults.' });
  };

  const updateZoneField = (i: number, field: keyof ZonePixel, value: string | number) => {
    const updated = [...zones];
    updated[i] = { ...updated[i], [field]: value };
    setZones(updated);
  };

  const toggleJunction = (junction: string) => {
    setActiveJunctions(prev =>
      prev.includes(junction) ? prev.filter(j => j !== junction) : [...prev, junction]
    );
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

          {/* Zone coordinates — pixel-based */}
          <div className="dashboard-card space-y-4">
            <h3 className="text-sm font-semibold text-foreground">Zone Coordinates</h3>
            <div className="flex items-start gap-2 p-2.5 rounded-md bg-primary/5 border border-primary/20">
              <Info className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
              <p className="text-xs text-primary">
                These pixel coordinates define the No-Parking rectangle on the 640×480 video frame.
              </p>
            </div>
            <div className="space-y-4">
              {zones.map((zone, i) => (
                <div key={i} className="space-y-2 p-3 rounded-lg border border-border/50 bg-accent/5">
                  <div className="space-y-1">
                    <Label className="text-xs">Zone Name</Label>
                    <Input
                      value={zone.name}
                      onChange={(e) => updateZoneField(i, 'name', e.target.value)}
                    />
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <div className="space-y-1">
                      <Label className="text-xs font-mono">x1</Label>
                      <Input
                        type="number"
                        min={0}
                        max={640}
                        value={zone.x1}
                        onChange={(e) => updateZoneField(i, 'x1', Math.min(640, Math.max(0, parseInt(e.target.value) || 0)))}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs font-mono">y1</Label>
                      <Input
                        type="number"
                        min={0}
                        max={480}
                        value={zone.y1}
                        onChange={(e) => updateZoneField(i, 'y1', Math.min(480, Math.max(0, parseInt(e.target.value) || 0)))}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs font-mono">x2</Label>
                      <Input
                        type="number"
                        min={0}
                        max={640}
                        value={zone.x2}
                        onChange={(e) => updateZoneField(i, 'x2', Math.min(640, Math.max(0, parseInt(e.target.value) || 0)))}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs font-mono">y2</Label>
                      <Input
                        type="number"
                        min={0}
                        max={480}
                        value={zone.y2}
                        onChange={(e) => updateZoneField(i, 'y2', Math.min(480, Math.max(0, parseInt(e.target.value) || 0)))}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Yatra Mode Configuration */}
          <div className="dashboard-card space-y-4">
            <h3 className="text-sm font-semibold text-foreground">Yatra Mode Configuration</h3>
            <p className="text-xs text-muted-foreground">
              Override signal timing during religious processions and large public events.
            </p>

            <div className="space-y-2">
              <Label htmlFor="override-time">Override Signal Time (seconds)</Label>
              <Input
                id="override-time"
                type="number"
                min={10}
                max={120}
                value={overrideSignalTime}
                onChange={(e) => setOverrideSignalTime(Math.min(120, Math.max(10, parseInt(e.target.value) || 45)))}
              />
              <p className="text-xs text-muted-foreground">
                All selected junctions will hold green for this duration during Yatra mode.
              </p>
            </div>

            <div className="space-y-2">
              <Label>Active Junctions</Label>
              <div className="space-y-2 p-3 rounded-lg border border-border/50 bg-accent/5">
                {allJunctions.map((junction) => (
                  <label key={junction} className="flex items-center gap-2.5 cursor-pointer">
                    <Checkbox
                      checked={activeJunctions.includes(junction)}
                      onCheckedChange={() => toggleJunction(junction)}
                    />
                    <span className="text-sm text-foreground">{junction}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="emergency-contact">Emergency Contact</Label>
              <Input
                id="emergency-contact"
                value={emergencyContact}
                onChange={(e) => setEmergencyContact(e.target.value)}
                placeholder="+91 98765 43210"
              />
              <p className="text-xs text-muted-foreground">
                Primary contact for Yatra mode activation alerts.
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 pb-6">
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
