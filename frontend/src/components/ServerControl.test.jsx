import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import ServerControl from './ServerControl';
import { I18nProvider } from '../i18n/I18nContext';

describe('ServerControl Component', () => {
  const mockStatus = {
    is_running: true,
    model: 'gemma4',
    device: 'gpu',
    slots: {
      llm: { model: 'gemma4', device: 'gpu', status: 'running', is_loaded: true },
      stt: { model: 'openai/whisper-base', device: 'cpu', status: 'stopped', is_loaded: false },
      tts: { model: 'Audio8/Audio8-TTS-Preview-0.1b', device: 'npu', status: 'stopped', is_loaded: false },
    },
    gpu_usage: { percent: 15.0 },
    npu_usage: { percent: 0.0 }
  };

  const mockModels = [
    { id: 'gemma4', name: 'Gemma 4 (12B)', task: 'text-generation', size_gb: 7.2 },
    { id: 'openai/whisper-base', name: 'Whisper Base', task: 'speech-to-text', size_gb: 0.15 },
    { id: 'Audio8/Audio8-TTS-Preview-0.1b', name: 'Audio8 TTS 0.1B', task: 'text-to-speech', size_gb: 0.2 }
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn().mockImplementation((url) => {
      if (url.includes('/api/kernels/status')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ compiled_models: [] })
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ success: true })
      });
    });
  });

  it('renders Tri-Hardware Master controller and 3 slot cards', () => {
    render(
      <I18nProvider>
        <ServerControl
          apiBase="http://localhost:8080"
          status={mockStatus}
          models={mockModels}
          modelsLoading={false}
          onRefresh={vi.fn()}
        />
      </I18nProvider>
    );

    // Master controller
    expect(screen.getByText(/Inference Engine Controller|Controller del Motore/i)).toBeTruthy();
    expect(screen.getByText(/Start Tri-Hardware Live|Avvia Tri-Hardware Live/i)).toBeTruthy();
    expect(screen.getByText(/Stop All|Arresta Tutti/i)).toBeTruthy();

    // 3 Dedicated slot cards
    expect(screen.getAllByText(/Slot LLM/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Slot STT/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Slot TTS/i).length).toBeGreaterThan(0);
  });

  it('allows selecting device for slots and triggers slot start', async () => {
    render(
      <I18nProvider>
        <ServerControl
          apiBase="http://localhost:8080"
          status={mockStatus}
          models={mockModels}
          modelsLoading={false}
          onRefresh={vi.fn()}
        />
      </I18nProvider>
    );

    // Find a Start Slot button
    const startButtons = screen.getAllByText(/Start Slot|Avvia Slot/i);
    expect(startButtons.length).toBeGreaterThan(0);

    fireEvent.click(startButtons[0]);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/control/slot/start'),
        expect.objectContaining({
          method: 'POST'
        })
      );
    });
  });

  it('triggers master live start when clicking Start Tri-Hardware Live', async () => {
    render(
      <I18nProvider>
        <ServerControl
          apiBase="http://localhost:8080"
          status={mockStatus}
          models={mockModels}
          modelsLoading={false}
          onRefresh={vi.fn()}
        />
      </I18nProvider>
    );

    const masterStartBtn = screen.getByText(/Start Tri-Hardware Live|Avvia Tri-Hardware Live/i);
    fireEvent.click(masterStartBtn);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/live/start'),
        expect.objectContaining({
          method: 'POST'
        })
      );
    });
  });
});
