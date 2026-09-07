import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import LiveStudio from './LiveStudio';
import { I18nProvider } from '../i18n/I18nContext';

describe('LiveStudio Component', () => {
  beforeEach(() => {
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
  });

  const mockStatus = {
    is_running: true,
    model: 'gemma3',
    device: 'gpu',
    slots: {
      llm: { model: 'gemma3', device: 'gpu', status: 'running' },
      stt: { model: 'openai/whisper-base', device: 'npu', status: 'running' },
      tts: { model: 'Audio8/Audio8-TTS-Preview-0.1b', device: 'cpu', status: 'running' },
    },
    gpu_usage: { percent: 12.5 },
    npu_usage: { percent: 0.0 }
  };

  it('renders title and Tri-Hardware engine cards', () => {
    render(
      <I18nProvider>
        <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
      </I18nProvider>
    );

    expect(screen.getByText(/Live Voice-to-Voice Studio/i)).toBeTruthy();
    expect(screen.getByText(/Tri-Hardware Engine/i)).toBeTruthy();
    expect(screen.getAllByText(/AMD Radeon 890M GPU/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/AMD Ryzen AI NPU/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/AMD Ryzen 24-Thread CPU/i).length).toBeGreaterThan(0);
  });

  it('renders latency HUD and start session button', () => {
    render(
      <I18nProvider>
        <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
      </I18nProvider>
    );

    expect(screen.getByText(/STT Latency/i)).toBeTruthy();
    expect(screen.getByText(/Time to First Token/i)).toBeTruthy();
    expect(screen.getByText(/Time to First Audio/i)).toBeTruthy();
    expect(screen.getByText(/E2E Turnaround/i)).toBeTruthy();
    expect(screen.getByText(/Start Live Session/i)).toBeTruthy();
  });

  it('renders insecure context warning and HTTPS button on LAN HTTP', () => {
    const originalLocation = window.location;
    const originalSecure = window.isSecureContext;

    delete window.location;
    window.location = new URL('http://192.168.132.197:8080/');
    window.isSecureContext = false;

    try {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://192.168.132.197:8080" status={mockStatus} />
        </I18nProvider>
      );

      expect(screen.getByRole('alert')).toBeTruthy();
      expect(screen.getByText(/Microphone Access Required/i)).toBeTruthy();

      const httpsLink = screen.getByRole('link', { name: /Switch to HTTPS/i });
      expect(httpsLink).toBeTruthy();
      expect(httpsLink.getAttribute('href')).toBe('https://192.168.132.197:8443');

      const toggleGuideBtn = screen.getByRole('button', { name: /Guide for Chrome Android/i });
      fireEvent.click(toggleGuideBtn);
      expect(screen.getByText(/chrome:\/\/flags\/#unsafely-treat-insecure-origin-as-secure/i)).toBeTruthy();
    } finally {
      window.location = originalLocation;
      window.isSecureContext = originalSecure;
    }
  });

  it('displays visible error banner when microphone access fails', async () => {
    class MockWebSocket {
      constructor() {
        setTimeout(() => {
          if (this.onopen) this.onopen();
        }, 10);
      }
      send() {}
      close() {}
    }
    const origWS = window.WebSocket;
    const origFetch = window.fetch;
    const origMediaDevices = navigator.mediaDevices;

    window.WebSocket = MockWebSocket;
    window.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });

    const permissionError = new Error('Permission denied');
    permissionError.name = 'NotAllowedError';
    navigator.mediaDevices = {
      getUserMedia: vi.fn().mockRejectedValue(permissionError)
    };

    try {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const startBtns = screen.getAllByRole('button', { name: /Start Live Session/i });
      fireEvent.click(startBtns[0]);

      const errorAlert = await screen.findByText(/Microphone access was denied/i);
      expect(errorAlert).toBeTruthy();
    } finally {
      window.WebSocket = origWS;
      window.fetch = origFetch;
      navigator.mediaDevices = origMediaDevices;
    }
  });
});
