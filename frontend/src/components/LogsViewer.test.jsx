import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import LogsViewer from './LogsViewer';
import { I18nProvider } from '../i18n/I18nContext';

describe('LogsViewer Component', () => {
  const mockApiBase = 'http://localhost:8080';
  const initialLogs = ['[2026-09-07] Server starting...', '[2026-09-07] NPU kernel initialized.'];
  const newLogs = [...initialLogs, '[2026-09-07] Model ready.'];
  const mockNpu = {
    device_node: true,
    xrt_smi: true,
    pyxrt_import: true
  };

  let scrollIntoViewMock;

  beforeEach(() => {
    vi.useFakeTimers();
    scrollIntoViewMock = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;

    global.fetch = vi.fn((url) => {
      if (url.includes('/api/logs')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ logs: initialLogs })
        });
      }
      if (url.includes('/api/npu/check')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockNpu)
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('renders NPU health diagnostics and initial logs', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <LogsViewer apiBase={mockApiBase} />
        </I18nProvider>
      );
    });

    // Check log lines are rendered
    expect(screen.getByText(/Server starting\.\.\./)).toBeDefined();
    expect(screen.getByText(/NPU kernel initialized\./)).toBeDefined();

    // Check autoScroll checkbox exists and is checked by default
    const autoScrollCheckbox = screen.getByRole('checkbox');
    expect(autoScrollCheckbox.checked).toBe(true);
  });

  it('executes scrollIntoView when autoScroll is true', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <LogsViewer apiBase={mockApiBase} />
        </I18nProvider>
      );
    });

    expect(scrollIntoViewMock).toHaveBeenCalled();
  });

  it('does NOT execute scrollIntoView when autoScroll is unchecked and preserves manual scroll position', async () => {
    let logsResponse = { logs: initialLogs };
    global.fetch = vi.fn((url) => {
      if (url.includes('/api/logs')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(logsResponse)
        });
      }
      if (url.includes('/api/npu/check')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockNpu)
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    let rendered;
    await act(async () => {
      rendered = render(
        <I18nProvider>
          <LogsViewer apiBase={mockApiBase} />
        </I18nProvider>
      );
    });

    scrollIntoViewMock.mockClear();

    // Uncheck autoScroll
    const autoScrollCheckbox = screen.getByRole('checkbox');
    await act(async () => {
      fireEvent.click(autoScrollCheckbox);
    });
    expect(autoScrollCheckbox.checked).toBe(false);

    // Simulate manual scroll
    const logContainer = rendered.container.querySelector('[role="log"]');
    expect(logContainer).toBeDefined();
    logContainer.scrollTop = 120;
    await act(async () => {
      fireEvent.scroll(logContainer);
    });

    // Clear calls before new logs arrive
    scrollIntoViewMock.mockClear();

    // Now new logs arrive via poll
    logsResponse = { logs: newLogs };
    await act(async () => {
      vi.advanceTimersByTime(2500);
    });

    // scrollIntoView must NOT have been called because autoScroll is false
    expect(scrollIntoViewMock).not.toHaveBeenCalled();

    // manual scroll position is preserved
    expect(logContainer.scrollTop).toBe(120);
  });

  it('does NOT poll logs when isActive is false', async () => {
    let fetchCount = 0;
    global.fetch = vi.fn((url) => {
      if (url.includes('/api/logs')) {
        fetchCount++;
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ logs: initialLogs })
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    await act(async () => {
      render(
        <I18nProvider>
          <LogsViewer apiBase={mockApiBase} isActive={false} />
        </I18nProvider>
      );
    });

    const initialCount = fetchCount;
    expect(initialCount).toBe(0);

    await act(async () => {
      vi.advanceTimersByTime(5000);
    });

    expect(fetchCount).toBe(0);
  });
});
