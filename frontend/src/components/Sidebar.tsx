import { NavLink } from 'react-router-dom';
import './Sidebar.css';

const NAV_ITEMS = [
  { to: '/',       icon: '⌂', label: 'Dashboard' },
  { to: '/upload', icon: '↑', label: 'Upload ECG' },
  { to: '/batch',  icon: '⊞', label: 'Batch Analysis' },
] as const;

export function Sidebar() {
  return (
    <aside className="sidebar" id="sidebar-nav">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <span className="sidebar-logo-icon">♥</span>
        </div>
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-name">Aortica</span>
          <span className="sidebar-brand-version">v0.2.0</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <ul className="sidebar-nav-list">
          {NAV_ITEMS.map(item => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `sidebar-nav-link ${isActive ? 'active' : ''}`
                }
                id={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
              >
                <span className="sidebar-nav-icon">{item.icon}</span>
                <span className="sidebar-nav-label">{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-footer-status">
          <span className="status-dot status-dot--online" />
          <span className="sidebar-footer-text">Server Online</span>
        </div>
      </div>
    </aside>
  );
}
