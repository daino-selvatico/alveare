import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import BenchmarksView from './BenchmarksView';
import { I18nProvider } from '../i18n/I18nContext';

describe('BenchmarksView', () => {
  const mockApiBase = 'http://127.0.0.1:8080';

  const mockModels = [
    {
      id: 'gemma-2b-it',
      alias: 'gemma2b',
      arch: 'gemma',
      size_mb: 2560,
      has_config: true,
      active: true,
      path: '/models/gemma-2b-it.gguf'
    },
    {
      id: 'llama-3.2-1b',
      alias: 'llama3',
      arch: 'llama',
      size_mb: 800,
      has_config: false,
      active: false,
      path: '/models/llama-3.2-1b.gguf'
    }
  ];

  const mockStatus = {
    is_running: true,
    model: 'gemma-2b-it',
    is_loaded: true,
    load_progress: 100,
    load_step: 'Ready',
    tok_per_sec: 14.85
  };

  beforeEach(() => {
    vi.useFakeTimers();
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
          json: () => Promise.resolve(mockModels)
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('renders benchmark header and metrics correctly', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <BenchmarksView
            apiBase={mockApiBase}
            status={mockStatus}
            models={mockModels}
          />
        </I18nProvider>
      );
    });

    expect(screen.getAllByText('gemma-2b-it').length).toBeGreaterThan(0);
    expect(screen.getByText(/14.85/)).toBeDefined();
    expect(screen.getByText('2.50 GB')).toBeDefined();
    expect(screen.getByText('800 MB')).toBeDefined();
  });

  it('polls /api/status and /api/models periodically', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <BenchmarksView
            apiBase={mockApiBase}
            status={mockStatus}
            models={mockModels}
          />
        </I18nProvider>
      );
    });

    const initialFetchCount = global.fetch.mock.calls.length;

    await act(async () => {
      vi.advanceTimersByTime(2100);
    });

    expect(global.fetch.mock.calls.length).toBeGreaterThan(initialFetchCount);
  });

  it('handles fetch error gracefully without crashing', async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error('Network error')));

    await act(async () => {
      render(
        <I18nProvider>
          <BenchmarksView
            apiBase={mockApiBase}
            status={null}
            models={[]}
          />
        </I18nProvider>
      );
    });

    expect(screen.getByRole('alert')).toBeDefined();
  });
});
