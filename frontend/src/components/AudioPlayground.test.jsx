import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import AudioPlayground from './AudioPlayground';
import { I18nProvider } from '../i18n/I18nContext';

describe('AudioPlayground', () => {
  const mockApiBase = 'http://127.0.0.1:8080';
  const mockStatus = { is_running: true, model: 'sensevoice' };

  it('renders AudioPlayground in realtime mode with mic button', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={mockStatus}
          activeModel="sensevoice"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    expect(screen.getByText(/SenseVoice Small STT Workbench/i)).toBeDefined();
    expect(screen.getByText(/Streaming Live \(Microfono\)/i)).toBeDefined();
    expect(screen.getByText(/File Audio/i)).toBeDefined();
  });

  it('switches to File Audio mode and displays drag-and-drop zone', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={mockStatus}
          activeModel="sensevoice"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    const fileTabBtn = screen.getByText(/File Audio/i);
    fireEvent.click(fileTabBtn);

    expect(screen.getByText(/Trascina o carica un file audio/i)).toBeDefined();
    expect(screen.getByText(/Supporta MP3, WAV, M4A, OGG, FLAC/i)).toBeDefined();
  });
});
