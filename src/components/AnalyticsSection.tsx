import { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Cell, Area, ComposedChart, Legend, LabelList,
  ReferenceArea,
} from 'recharts';

// Generate RL learning curve data once — Day 1 forced to 80
const rlData = Array.from({ length: 30 }, (_, i) => ({
  day: `Day ${i + 1}`,
  waitTime: i === 0 ? 80 : Math.round(25 + 55 * Math.exp(-0.1 * (i + 1)) + (Math.random() * 6 - 3)),
}));

const slotsEstimate: Record<string, number> = {
  'Navi Peth': 320,
  'Chatti Galli': 230,
  'Station Road': 200,
  'Hutatma Chowk': 150,
  'Textile Market': 110,
};

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
  const slots = slotsEstimate[d.zone] ?? 100;
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 text-xs space-y-1">
      <p className="text-foreground font-medium">{d.zone}</p>
      <p className="text-muted-foreground">Violations: {d.incidents}/month</p>
      <p style={{ color: severityColor[d.severity] }}>Severity: {d.severity}</p>
      <p className="text-primary text-[10px]">Recommendation: Build multi-level parking — Est. {slots} slots needed</p>
    </div>
  );
};

const rlLegend = () => (
  <div className="flex items-center gap-4 justify-center mt-1">
    <span className="flex items-center gap-1.5 text-[10px] text-primary">
      <span className="w-4 h-0.5 bg-primary inline-block rounded" /> RL Wait Time
    </span>
    <span className="flex items-center gap-1.5 text-[10px] text-warning">
      <span className="w-4 h-0.5 border-t border-dashed border-warning inline-block" /> Legacy Fixed Signal (60s baseline)
    </span>
    <span className="flex items-center gap-1.5 text-[10px] text-success">
      <span className="w-3 h-3 bg-success/20 inline-block rounded-sm" /> AI-Optimized Zone
    </span>
  </div>
);

export function AnalyticsSection() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* RL Learning Curve */}
      <div className="dashboard-card">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-foreground">RL-Optimizer: 30-Day Learning Progress</h3>
          <div className="flex items-center gap-2">
            <span className="text-[10px] bg-success/20 text-success px-2 py-0.5 rounded-full font-mono font-medium">
              ↓ 68.7% Wait Time Reduction
            </span>
            <span className="text-[10px] bg-primary/20 text-primary px-2 py-0.5 rounded-full font-mono">
              30-Day Simulation
            </span>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mb-4">Average vehicle wait time at Market Yard Junction</p>
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={rlData}>
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
            <ReferenceArea x1="Day 18" x2="Day 30" y1={0} y2={100} fill="hsl(142 71% 45%)" fillOpacity={0.07} />
            <ReferenceLine
              y={60}
              stroke="#f59e0b"
              strokeDasharray="6 3"
            />
            <Area
              type="monotone"
              dataKey="waitTime"
              fill="hsl(189 94% 53% / 0.08)"
              stroke="none"
            />
            <Line
              type="monotone"
              dataKey="waitTime"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#22d3ee' }}
            />
          </ComposedChart>
        </ResponsiveContainer>
        {rlLegend()}
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
              label={{ value: 'Violations / Month', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#9ca3af' }}
            />
            <Tooltip content={<CustomTooltipParking />} />
            <Bar dataKey="incidents" radius={[4, 4, 0, 0]}>
              <LabelList dataKey="incidents" position="top" fill="#9ca3af" fontSize={10} />
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
