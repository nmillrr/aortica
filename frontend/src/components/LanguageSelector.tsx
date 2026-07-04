import { useTranslation } from 'react-i18next';
import './LanguageSelector.css';

const LANGUAGES = ['en', 'fr', 'es', 'sw'] as const;
type LanguageCode = (typeof LANGUAGES)[number];

/**
 * Header language selector. Switches the active i18next language at runtime
 * (no page reload); the choice is persisted to localStorage by i18next's
 * LanguageDetector (key: `aortica_language`).
 */
export function LanguageSelector() {
  const { t, i18n } = useTranslation();
  const current = (i18n.resolvedLanguage ?? 'en') as LanguageCode;

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    void i18n.changeLanguage(event.target.value);
  };

  return (
    <label className="language-selector" title={t('languageSelector.label')}>
      <span className="language-selector-icon" aria-hidden="true">🌐</span>
      <span className="sr-only">{t('languageSelector.label')}</span>
      <select
        className="language-selector-select"
        id="language-selector"
        value={current}
        onChange={handleChange}
        aria-label={t('languageSelector.label')}
      >
        {LANGUAGES.map(code => (
          <option key={code} value={code}>
            {t(`languageSelector.${code}`)}
          </option>
        ))}
      </select>
    </label>
  );
}
