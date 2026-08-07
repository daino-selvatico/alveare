import React, { createContext, useContext, useState, useCallback } from 'react';

import en from './en.json';
import it from './it.json';

const translations = { en, it };
const STORAGE_KEY = 'alveare_language';

export const I18nContext = createContext(null);


function detectDefaultLanguage() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === 'en' || saved === 'it') {
    return saved;
  }
  const browserLang = (navigator.language || navigator.userLanguage || '').toLowerCase();
  if (browserLang.startsWith('it')) {
    return 'it';
  }
  return 'en';
}

function getNestedValue(obj, path) {
  if (!obj || !path) return undefined;
  const parts = path.split('.');
  let current = obj;
  for (const part of parts) {
    if (current && typeof current === 'object' && part in current) {
      current = current[part];
    } else {
      return undefined;
    }
  }
  return current;
}

export function I18nProvider({ children }) {
  const [language, setLanguageState] = useState(detectDefaultLanguage);

  const setLanguage = useCallback((lang) => {
    const validLang = lang === 'it' ? 'it' : 'en';
    setLanguageState(validLang);
    try {
      localStorage.setItem(STORAGE_KEY, validLang);
    } catch (e) {
      console.error('Failed to save language setting:', e);
    }
  }, []);

  const t = useCallback((key, params = {}) => {
    let template = getNestedValue(translations[language], key);
    if (template === undefined) {
      template = getNestedValue(translations.en, key);
    }
    if (template === undefined) {
      return key;
    }
    if (typeof template !== 'string') {
      return template;
    }

    return template.replace(/\{{?(\w+)\}?}/g, (match, paramName) => {
      return params[paramName] !== undefined ? params[paramName] : match;
    });
  }, [language]);

  return (
    <I18nContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useTranslation() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useTranslation must be used within an I18nProvider');
  }
  return context;
}
