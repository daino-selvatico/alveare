import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import App from './App';
import { I18nProvider } from './i18n/I18nContext';

describe('App Polling and Tab Navigation', () => {
  const mockStatus = {
    is_running: true,
    model: 'gemma4-e4b',
    device: 'gpu',
    slots: {
      llm: { model: 'gemma4-e4b', device: 'gpu', status: 'running' },
      stt: { model: 'openai/whisper-large-v3-turbo', device: 'cpu', status: 'running' },
      tts: { model: 'hexgrad/Kokoro-82M', device: 'cpu', status: 'running' },
    }
  };

  beforeEach(() => {
    vi.useFakeTimers();
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      arc: vi.fn(),
      rect: vi.fn(),
      roundRect: vi.fn(),
      createLinearGradient: vi.fn(() => ({
        addColorStop: vi.fn(),
      })),
    }));
    global.fetch = vi.fn((url) => {
      if (url.includes('/api/status')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockStatus)
        });
      }
      if (url.includes('/api/models')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([])
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('renders navbar and switches to Live Studio tab', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <App />
        </I18nProvider>
      );
    });

    // Default tab is chat
    const liveTabBtn = screen.getByRole('tab', { name: /Live Studio/i });
    expect(liveTabBtn).toBeDefined();

    await act(async () => {
      fireEvent.click(liveTabBtn);
    });

    const livePanel = document.getElementById('panel-live');
    expect(livePanel.style.display).toBe('flex');
  });

  it('pauses polling when tab is hidden and resumes when visible', async () => {
    let fetchStatusCount = 0;
    global.fetch = vi.fn((url) => {
      if (url.includes('/api/status')) {
        fetchStatusCount++;
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockStatus)
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
    });

    await act(async () => {
      render(
        <I18nProvider>
          <App />
        </I18nProvider>
      );
    });

    const initialFetchCount = fetchStatusCount;
    expect(initialFetchCount).toBeGreaterThan(0);

    // Simulate tab hidden
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'hidden'
    });
    document.dispatchEvent(new Event('visibilitychange'));

    // Advance 6 seconds while hidden
    await act(async () => {
      vi.advanceTimersByTime(6000);
    });

    const countWhileHidden = fetchStatusCount;
    // Should NOT have polled during hidden interval
    expect(countWhileHidden).toBe(initialFetchCount);

    // Simulate tab visible again
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'visible'
    });
    document.dispatchEvent(new Event('visibilitychange'));

    expect(fetchStatusCount).toBeGreaterThan(countWhileHidden);
  });
});
