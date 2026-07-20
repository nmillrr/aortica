import { NavLink } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ConnectionStatusBanner } from './ConnectionStatusBanner';
import './Sidebar.css';

const NAV_ITEMS = [
  { to: '/',              icon: '⌂', id: 'dashboard',    labelKey: 'nav.dashboard' },
  { to: '/upload',        icon: '↑', id: 'upload',       labelKey: 'nav.uploadEcg' },
  { to: '/history',       icon: '📋', id: 'history',      labelKey: 'nav.ecgHistory' },
  { to: '/worklist',      icon: '⚑', id: 'worklist',     labelKey: 'nav.worklist' },
  { to: '/batch',         icon: '⊞', id: 'batch',        labelKey: 'nav.batchAnalysis' },
  { to: '/report-event',  icon: '⚠', id: 'report-event', labelKey: 'nav.reportEvent' },
  { to: '/admin',         icon: '⚙', id: 'admin',        labelKey: 'nav.admin' },
  { to: '/federated',     icon: '🌐', id: 'federated',    labelKey: 'nav.federated' },
  { to: '/validation/sites', icon: '🌍', id: 'validation', labelKey: 'nav.validation' },
  { to: '/validation/prospective', icon: '🧪', id: 'prospective', labelKey: 'nav.prospective' },
  { to: '/validation/monitor', icon: '📈', id: 'monitor', labelKey: 'nav.monitor' },
  { to: '/analytics/sites', icon: '🗺', id: 'siteAnalytics', labelKey: 'nav.siteAnalytics' },
  { to: '/compare',       icon: '⇄', id: 'compare',      labelKey: 'nav.compare' },
] as const;

export function Sidebar() {
  const { t } = useTranslation();

  return (
    <aside className="sidebar" id="sidebar-nav">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <span className="sidebar-logo-icon">♥</span>
        </div>
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-name">{t('common.appName')}</span>
          <span className="sidebar-brand-version">{t('common.version')}</span>
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
                id={`nav-${item.id}`}
              >
                <span className="sidebar-nav-icon">{item.icon}</span>
                <span className="sidebar-nav-label">{t(item.labelKey)}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer with live connection status */}
      <div className="sidebar-footer">
        <ConnectionStatusBanner />
      </div>
    </aside>
  );
}
