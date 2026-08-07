import React from 'react';
import { useTranslation } from '../i18n/I18nContext';

export default function LanguageSwitcher() {
  const { language, setLanguage } = useTranslation();

  return (
    <div
      role="group"
      aria-label="Language Selector"
      style={{
        display: 'flex',
        alignItems: 'center',
        background: 'rgba(0, 0, 0, 0.25)',
        padding: '0.2rem',
        borderRadius: '8px',
        border: '1px solid var(--border-color)'
      }}
    >
      <button
        onClick={() => setLanguage('it')}
        aria-pressed={language === 'it'}
        style={{
          padding: '0.25rem 0.55rem',
          fontSize: '0.78rem',
          fontWeight: 700,
          borderRadius: '6px',
          border: 'none',
          background: language === 'it' ? 'var(--gradient-brand)' : 'transparent',
          color: language === 'it' ? 'white' : 'var(--text-muted)',
          cursor: 'pointer',
          transition: 'all 0.2s ease'
        }}
        title="Italiano"
        aria-label="Imposta lingua Italiano"
      >
        IT
      </button>
      <button
        onClick={() => setLanguage('en')}
        aria-pressed={language === 'en'}
        style={{
          padding: '0.25rem 0.55rem',
          fontSize: '0.78rem',
          fontWeight: 700,
          borderRadius: '6px',
          border: 'none',
          background: language === 'en' ? 'var(--gradient-brand)' : 'transparent',
          color: language === 'en' ? 'white' : 'var(--text-muted)',
          cursor: 'pointer',
          transition: 'all 0.2s ease'
        }}
        title="English"
        aria-label="Set language English"
      >
        EN
      </button>
    </div>
  );
}
