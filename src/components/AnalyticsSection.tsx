import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Cell,
} from 'recharts';

// Generate RL learning curve data once
const rlData = Array.from({ length: 30 }, (_, i) => ({
  day: `Day ${i + 1}`,
  waitTime: Math.round(25 + 55 * Math.exp(-0.1 * (i + 1)) + (Math.random() * 6 - 3)),
}));

const parkingData = [
  { zone: 'Navi Peth', incidents: 847, severity: 'Critical' },
  { zone: 'Chatti Galli', incidents: 612, severity: 'High' },
  { zone: 'Station Road', incidents: 534, severity: 'High' },
  { zone: 'Hutatma Chowk', incidents: 389, severity: 'Medium' },
  { zone: 'Textile Market', incidents: 298, severity: 'Medium' },
];

const severityColor: Record<string, string> = {
  Critical: '#ef4444',
  High: '#f97316',
  Medium: '#eab308',
};

const CustomTooltipRL = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 text-xs">
      <p className="text-foreground font-medium">{label}</p>
      <p className="text-primary">Wait Time: {payload[0].value}s</p>
    </div>
  );
};

const CustomTooltipParking = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 text-xs space-y-1">
      <p className="text-foreground font-medium">{d.zone}</p>
      <p className="text-muted-foreground">Incidents: {d.incidents}/month</p>
      <p style={{ color: severityColor[d.severity] }}>Severity: {d.severity}</p>
      <p className="text-primary text-[10px]">Recommended: Build municipal parking lot here</p>
    </div>
  );
};

export function AnalyticsSection() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* RL Learning Curve */}
      <div className="dashboard-card">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-foreground">RL-Optimizer: 30-Day Learning Progress</h3>
          <span className="text-[10px] bg-primary/20 text-primary px-2 py-0.5 rounded-full font-mono">
            30-Day Simulation
          </span>
        </div>
        <p className="text-xs text-muted-foreground mb-4">Average vehicle wait time at Market Yard Junction</p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={rlData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 17%)" />
            <XAxis
              dataKey="day"
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickLine={false}
              axisLine={{ stroke: 'hsl(217 33% 17%)' }}
              interval={4}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickLine={false}
              axisLine={{ stroke: 'hsl(217 33% 17%)' }}
              label={{ value: 'Wait Time (s)', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#9ca3af' }}
            />
            <Tooltip content={<CustomTooltipRL />} />
            <ReferenceLine
              y={60}
              stroke="#f59e0b"
              strokeDasharray="6 3"
              label={{ value: 'Old fixed-timer baseline', fill: '#f59e0b', fontSize: 10, position: 'right' }}
            />
            <Line
              type="monotone"
              dataKey="waitTime"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#22d3ee' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Ghost Parking */}
      <div className="dashboard-card">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-foreground">Illegal Parking Demand by Zone</h3>
        </div>
        <p className="text-xs text-muted-foreground mb-4">Recommends high-ROI municipal parking lot locations</p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={parkingData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 17%)" />
            <XAxis
              dataKey="zone"
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickLine={false}
              axisLine={{ stroke: 'hsl(217 33% 17%)' }}
              angle={-20}
              textAnchor="end"
              height={60}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickLine={false}
              axisLine={{ stroke: 'hsl(217 33% 17%)' }}
              label={{ value: 'Incidents / Month', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#9ca3af' }}
            />
            <Tooltip content={<CustomTooltipParking />} />
            <Bar dataKey="incidents" radius={[4, 4, 0, 0]}>
              {parkingData.map((entry, i) => (
                <Cell key={i} fill={severityColor[entry.severity]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-[10px] text-muted-foreground mt-2">● Data-driven SMC infrastructure recommendation</p>
      </div>
    </div>
  );
}
