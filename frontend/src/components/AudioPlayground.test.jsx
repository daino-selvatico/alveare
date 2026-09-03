import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import AudioPlayground from './AudioPlayground';
import { I18nProvider } from '../i18n/I18nContext';

describe('AudioPlayground', () => {
  const mockApiBase = 'http://127.0.0.1:8080';
  const mockStatus = { is_running: true, model: 'whisper-base' };

  it('renders AudioPlayground in realtime mode with mic button', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={mockStatus}
          activeModel="whisper-base"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    expect(screen.getByText(/Whisper STT Workbench/i)).toBeDefined();
    expect(screen.getByText(/Live Streaming|Streaming Live/i)).toBeDefined();
    expect(screen.getByText(/Audio File|File Audio/i)).toBeDefined();
  });

  it('switches to File Audio mode and displays drag-and-drop zone', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={mockStatus}
          activeModel="whisper-base"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    const fileTabBtn = screen.getByText(/Audio File|File Audio/i);
    fireEvent.click(fileTabBtn);

    expect(screen.getByText(/Trascina o carica un file audio|Drag & drop or upload/i)).toBeDefined();
    expect(screen.getByText(/Supporta MP3, WAV, M4A, OGG, FLAC|Supports MP3, WAV, M4A, OGG, FLAC/i)).toBeDefined();
  });

  it('switches to TTS mode and displays speech generation controls', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={mockStatus}
          activeModel="audio8-0.1b"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    const ttsTabBtn = screen.getByText(/Text to Speech|Sintesi Vocale/i);
    fireEvent.click(ttsTabBtn);

    expect(screen.getByText(/Testo da Sintetizzare|Text to Synthesize/i)).toBeDefined();
    expect(screen.getByText(/Genera Audio|Generate Speech/i)).toBeDefined();
  });

  it('switches to TTS mode and supports NPU/CPU device toggle', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={mockStatus}
          activeModel="audio8-0.1b"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    const ttsTabBtn = screen.getByText(/Text to Speech|Sintesi Vocale/i);
    fireEvent.click(ttsTabBtn);

    const npuToggle = screen.getByText(/⚡ NPU/i);
    const cpuToggle = screen.getByText(/🖥️ CPU/i);
    expect(npuToggle).toBeDefined();
    expect(cpuToggle).toBeDefined();

    // Switch to CPU
    fireEvent.click(cpuToggle);
    expect(cpuToggle.style.color).toBe('rgb(96, 165, 250)'); // #60a5fa

    // Switch back to NPU
    fireEvent.click(npuToggle);
    expect(npuToggle.style.color).toBe('rgb(52, 211, 153)'); // #34d399
  });

  it('renders disabled state when server is not running', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={{ is_running: false, model: 'whisper-base' }}
          activeModel="whisper-base"
          isServerRunning={false}
        />
      </I18nProvider>
    );

    expect(screen.getByText(/STT Server Not Running|Server STT Non Avviato/i)).toBeDefined();
  });
});
