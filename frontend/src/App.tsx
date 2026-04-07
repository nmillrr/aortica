import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { Results } from './pages/Results';
import { Batch } from './pages/Batch';
import { Login } from './pages/Login';

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/results/:id" element={<Results />} />
        <Route path="/batch" element={<Batch />} />
      </Route>
    </Routes>
  );
}
