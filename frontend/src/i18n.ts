import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import en from './locales/en/translation.json';
import fr from './locales/fr/translation.json';
import es from './locales/es/translation.json';
import sw from './locales/sw/translation.json';

const resources = {
  en: { translation: en },
  fr: { translation: fr },
  es: { translation: es },
  sw: { translation: sw },
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    supportedLngs: ['en', 'fr', 'es', 'sw'],
    interpolation: {
      escapeValue: false, // React already escapes
    },
    detection: {
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: 'aortica_language',
      caches: ['localStorage'],
    },
  });

/**
 * Keep the document's `lang` and `dir` attributes in sync with the active
 * language. `dir` is derived from i18next so that adding an RTL language
 * (Arabic/Hebrew) later flips the layout automatically — component styles use
 * CSS logical properties so this stub is all that's required.
 */
function applyDocumentLanguage(lng: string): void {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = lng;
  document.documentElement.dir = i18n.dir(lng);
}

applyDocumentLanguage(i18n.resolvedLanguage ?? 'en');
i18n.on('languageChanged', applyDocumentLanguage);

export default i18n;
