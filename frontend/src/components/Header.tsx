import { useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { LanguageSelector } from './LanguageSelector';
import './Header.css';

const PAGE_TITLE_KEYS: Record<string, string> = {
  '/':       'header.dashboard',
  '/upload': 'header.uploadEcg',
  '/batch':  'header.batchAnalysis',
};

export function Header() {
  const { pathname } = useLocation();
  const { t } = useTranslation();

  const titleKey = pathname.startsWith('/results/')
    ? 'header.analysisResults'
    : PAGE_TITLE_KEYS[pathname];

  const title = titleKey ? t(titleKey) : t('common.appName');

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
            placeholder={t('common.search')}
            id="global-search"
          />
        </div>
        <LanguageSelector />
        <button className="header-avatar btn-ghost" id="user-menu-btn" aria-label={t('header.userMenu')}>
          <span className="header-avatar-initials">A</span>
        </button>
      </div>
    </header>
  );
}
