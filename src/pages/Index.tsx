import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { VideoFeed } from '@/components/VideoFeed';
import { MetricCards } from '@/components/MetricCards';
import { AlertBanner } from '@/components/AlertBanner';
import { AnalyticsSection } from '@/components/AnalyticsSection';
import { AIChatWidget } from '@/components/AIChatWidget';
import { useTrafficData } from '@/hooks/useTrafficData';

const Index = () => {
  const {
    vehicleCount,
    encroachmentAlert,
    dynamicGreenTime,
    carbonSavedKg,
    backendStatus,
    error,
  } = useTrafficData();

  return (
    <div className="flex min-h-screen bg-background">
      <DashboardSidebar />
      <main className="flex-1 ml-[220px] transition-all duration-300 min-h-screen flex flex-col">
        <DashboardHeader />
        <div className="flex-1 p-6 space-y-4 overflow-y-auto">
          <AlertBanner encroachmentAlert={encroachmentAlert} error={error} />
          <MetricCards
            vehicleCount={vehicleCount}
            dynamicGreenTime={dynamicGreenTime}
            carbonSavedKg={carbonSavedKg}
            backendStatus={backendStatus}
            error={error}
          />
          <VideoFeed />
          <AnalyticsSection />
        </div>
      </main>
    </div>
  );
};

export default Index;
