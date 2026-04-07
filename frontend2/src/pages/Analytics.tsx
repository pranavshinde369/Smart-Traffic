import { DashboardSidebar } from '@/components/DashboardSidebar';
import { DashboardHeader } from '@/components/DashboardHeader';
import { AnalyticsSection } from '@/components/AnalyticsSection';

const Analytics = () => {
  return (
    <div className="flex min-h-screen bg-background">
      <DashboardSidebar />
      <main className="flex-1 ml-[220px] transition-all duration-300 min-h-screen flex flex-col">
        <DashboardHeader />
        <div className="flex-1 p-6 overflow-y-auto">
          <AnalyticsSection />
        </div>
      </main>
    </div>
  );
};

export default Analytics;
