import React from 'react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import ChatPlayground from './ChatPlayground';
import { I18nProvider } from '../i18n/I18nContext';

describe('ChatPlayground', () => {
  const mockApiBase = 'http://127.0.0.1:8080';
  const mockModels = [{ id: 'gemma4', name: 'Gemma 4' }];
  const mockStatus = { is_running: true, model: 'gemma4', tok_per_sec: 42.5 };

  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
    global.fetch = vi.fn((url) => {
      if (url.includes('/api/status')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockStatus)
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders ChatPlayground and gates image/audio uploads as coming soon', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <ChatPlayground
            apiBase={mockApiBase}
            status={mockStatus}
            activeModel="gemma4"
            isServerRunning={true}
            models={mockModels}
          />
        </I18nProvider>
      );
    });

    const attachBtn = screen.getByLabelText('Attach file (Documents)');
    expect(attachBtn).toBeDefined();

    await act(async () => {
      fireEvent.click(attachBtn);
    });

    const comingSoonBadges = screen.getAllByText('Coming soon');
    expect(comingSoonBadges.length).toBe(2);
  });

  it('rejects image files when uploaded', async () => {
    await act(async () => {
      render(
        <I18nProvider>
          <ChatPlayground
            apiBase={mockApiBase}
            status={mockStatus}
            activeModel="gemma4"
            isServerRunning={true}
            models={mockModels}
          />
        </I18nProvider>
      );
    });

    const region = screen.getByRole('region', { name: 'Area Playground Chat Multimodale' });
    const imageFile = new File(['fake image data'], 'photo.png', { type: 'image/png' });

    await act(async () => {
      fireEvent.drop(region, {
        dataTransfer: { files: [imageFile] }
      });
    });

    expect(await screen.findByText(/photo\.png/i)).toBeDefined();
  });
});
