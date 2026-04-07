import { useLocation } from 'react-router-dom';
import './Header.css';

const PAGE_TITLES: Record<string, string> = {
  '/':       'Dashboard',
  '/upload': 'Upload ECG',
  '/batch':  'Batch Analysis',
};

export function Header() {
  const { pathname } = useLocation();

  const title = pathname.startsWith('/results/')
    ? 'Analysis Results'
    : PAGE_TITLES[pathname] ?? 'Aortica';

  return (
    <header className="header glass" id="app-header">
      <div className="header-left">
        <h1 className="header-title">{title}</h1>
      </div>
      <div className="header-right">
        <div className="header-search">
          <span className="header-search-icon">⌕</span>
          <input
            type="text"
            className="header-search-input"
            placeholder="Search patients, ECGs..."
            id="global-search"
          />
        </div>
        <button className="header-avatar btn-ghost" id="user-menu-btn" aria-label="User menu">
          <span className="header-avatar-initials">A</span>
        </button>
      </div>
    </header>
  );
}
