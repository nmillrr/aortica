import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import './Layout.css';

export function Layout() {
  return (
    <div className="layout" id="app-layout">
      <Sidebar />
      <div className="layout-main">
        <Header />
        <main className="layout-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
