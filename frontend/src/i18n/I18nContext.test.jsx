import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { I18nProvider, useTranslation } from './I18nContext';

const TestComponent = () => {
  const { language, setLanguage, t } = useTranslation();

  return (
    <div>
      <span data-testid="current-language">{language}</span>
      <span data-testid="translated-light">{t('nav.themeLight')}</span>
      <span data-testid="translated-param">{t('nav.serverActive', { model: 'Llama-3' })}</span>
      <span data-testid="translated-fallback">{t('nonExistentKey')}</span>
      <button data-testid="btn-en" onClick={() => setLanguage('en')}>
        EN
      </button>
      <button data-testid="btn-it" onClick={() => setLanguage('it')}>
        IT
      </button>
    </div>
  );
};

describe('I18nContext & useTranslation', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('throws error when useTranslation is used outside I18nProvider', () => {
    // Suppress expected console error from React context missing
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => render(<TestComponent />)).toThrow(
      'useTranslation must be used within an I18nProvider'
    );

    consoleError.mockRestore();
  });

  it('detects language from localStorage if present', () => {
    localStorage.setItem('alveare_language', 'it');

    render(
      <I18nProvider>
        <TestComponent />
      </I18nProvider>
    );

    expect(screen.getByTestId('current-language').textContent).toBe('it');
    expect(screen.getByTestId('translated-light').textContent).toBe('Chiaro');
  });

  it('resolves key in default detected language (en when no localStorage)', () => {
    render(
      <I18nProvider>
        <TestComponent />
      </I18nProvider>
    );

    expect(screen.getByTestId('current-language').textContent).toBe('en');
    expect(screen.getByTestId('translated-light').textContent).toBe('Light');
  });

  it('switches language and persists selection in localStorage', () => {
    render(
      <I18nProvider>
        <TestComponent />
      </I18nProvider>
    );

    // Initial language is EN
    expect(screen.getByTestId('translated-light').textContent).toBe('Light');

    // Switch to IT
    act(() => {
      screen.getByTestId('btn-it').click();
    });

    expect(screen.getByTestId('current-language').textContent).toBe('it');
    expect(screen.getByTestId('translated-light').textContent).toBe('Chiaro');
    expect(localStorage.getItem('alveare_language')).toBe('it');

    // Switch back to EN
    act(() => {
      screen.getByTestId('btn-en').click();
    });

    expect(screen.getByTestId('current-language').textContent).toBe('en');
    expect(screen.getByTestId('translated-light').textContent).toBe('Light');
    expect(localStorage.getItem('alveare_language')).toBe('en');
  });

  it('interpolates parameters into translation string', () => {
    render(
      <I18nProvider>
        <TestComponent />
      </I18nProvider>
    );

    expect(screen.getByTestId('translated-param').textContent).toBe('Server Active (Llama-3)');

    act(() => {
      screen.getByTestId('btn-it').click();
    });

    expect(screen.getByTestId('translated-param').textContent).toBe('Server Attivo (Llama-3)');
  });

  it('falls back to raw key if key is missing', () => {
    render(
      <I18nProvider>
        <TestComponent />
      </I18nProvider>
    );

    expect(screen.getByTestId('translated-fallback').textContent).toBe('nonExistentKey');
  });
});
