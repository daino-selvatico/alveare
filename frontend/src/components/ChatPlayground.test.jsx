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

  it('renders ChatPlayground and provides image, audio, and document upload options', async () => {
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

    expect(screen.getByText('Image')).toBeDefined();
    expect(screen.getByText('Audio')).toBeDefined();
    expect(screen.getByText('Document')).toBeDefined();
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

  it('polls /api/status during streaming and displays server tok/s without ~', async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes('/api/status')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ is_running: true, model: 'gemma4', tok_per_sec: 3.6 })
        });
      }
      if (url.includes('/v1/chat/completions')) {
        const stream = new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'));
            controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":" world"}}]}\n\n'));
            controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
            controller.close();
          }
        });
        return Promise.resolve({ ok: true, body: stream });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    await act(async () => {
      render(
        <I18nProvider>
          <ChatPlayground
            apiBase={mockApiBase}
            status={{ tok_per_sec: 3.6 }}
            activeModel="gemma4"
            isServerRunning={true}
            models={mockModels}
          />
        </I18nProvider>
      );
    });

    const textarea = screen.getByLabelText('Campo testo messaggio');
    fireEvent.change(textarea, { target: { value: 'Test prompt' } });

    const sendBtn = screen.getByRole('button', { name: 'Send' });
    await act(async () => {
      fireEvent.click(sendBtn);
    });

    // Should display exact server rate 3.6 tok/s (not approximate with ~)
    expect(await screen.findByText(/3\.6 tok\/s/)).toBeDefined();
    expect(screen.queryByText(/~3\.6 tok\/s/)).toBeNull();
  });
});
