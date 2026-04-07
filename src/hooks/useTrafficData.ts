import { useState, useEffect, useRef, useCallback } from 'react';

interface TrafficData {
  vehicleCount: number;
  encroachmentAlert: boolean;
  dynamicGreenTime: number;
  carbonSavedKg: number;
  backendStatus: string;
  congestionPrediction: string;
  isLoading: boolean;
  error: string | null;
}

export function useTrafficData(): TrafficData {
  const [data, setData] = useState<TrafficData>({
    vehicleCount: 0,
    encroachmentAlert: false,
    dynamicGreenTime: 0,
    carbonSavedKg: 0,
    backendStatus: 'offline',
    congestionPrediction: '',
    isLoading: true,
    error: null,
  });
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/api/traffic-data');
      if (!res.ok) throw new Error('Backend error');
      const json = await res.json();
      setData({
        vehicleCount: json.vehicle_count ?? 0,
        encroachmentAlert: json.encroachment_alert ?? false,
        dynamicGreenTime: json.dynamic_green_time ?? 0,
        carbonSavedKg: json.carbon_saved_kg ?? 0,
        backendStatus: json.backend_status ?? 'offline',
        congestionPrediction: json.congestion_prediction ?? '',
        isLoading: false,
        error: null,
      });
    } catch {
      setData(prev => ({
        ...prev,
        isLoading: false,
        error: 'Backend unreachable',
        backendStatus: 'offline',
      }));
    }
  }, []);

  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(fetchData, 1000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchData]);

  return data;
}
