import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { ProtectedRoute } from './components/ProtectedRoute';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { Results } from './pages/Results';
import { ResultBrowser } from './pages/ResultBrowser';
import { Batch } from './pages/Batch';
import { Login } from './pages/Login';
import { AdverseEventWizard } from './pages/AdverseEventWizard';
import { AdverseEventHistory } from './pages/AdverseEventHistory';
import { AdminDashboard } from './pages/AdminDashboard';
import { FLDashboard } from './pages/FLDashboard';
import { SiteValidationPage } from './pages/SiteValidation';
import { ModelComparePage } from './pages/ModelCompare';
import { WorklistDashboard } from './pages/WorklistDashboard';
import { ReportPage } from './pages/ReportPage';
import { ProspectiveDataPage } from './pages/ProspectiveData';
import { PerformanceMonitorPage } from './pages/PerformanceMonitor';

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/history" element={<ResultBrowser />} />
        <Route path="/results/:id" element={<Results />} />
        <Route path="/batch" element={<Batch />} />
        <Route path="/report-event" element={<AdverseEventWizard />} />
        <Route path="/report-event/history" element={<AdverseEventHistory />} />
        <Route path="/admin" element={<AdminDashboard />} />
        <Route path="/federated" element={<FLDashboard />} />
        <Route path="/validation/sites" element={<SiteValidationPage />} />
        <Route path="/compare" element={<ModelComparePage />} />
        <Route path="/worklist" element={<WorklistDashboard />} />
        <Route path="/reports/:result_id" element={<ReportPage />} />
        <Route path="/validation/prospective" element={<ProspectiveDataPage />} />
        <Route path="/validation/monitor" element={<PerformanceMonitorPage />} />
      </Route>
    </Routes>
  );
}

