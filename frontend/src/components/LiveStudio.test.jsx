import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import LiveStudio, { downsampleTo16k, base64ToArrayBuffer, decodeAudioBuffer } from './LiveStudio';
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
    model: 'gemma4-e4b',
    device: 'gpu',
    slots: {
      llm: { model: 'gemma4-e4b', device: 'gpu', status: 'running' },
      stt: { model: 'openai/whisper-large-v3-turbo', device: 'cpu', status: 'running' },
      tts: { model: 'hexgrad/Kokoro-82M', device: 'cpu', status: 'running' },
    },
    gpu_usage: { percent: 12.5 },
    npu_usage: { percent: 0.0 }
  };

  it('renders correctly with slots and status indicators', () => {
    render(
      <I18nProvider>
        <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
      </I18nProvider>
    );

    expect(screen.getByText(/Live Voice-to-Voice Studio/i)).toBeTruthy();
    expect(screen.getByText('gemma4-e4b')).toBeTruthy();
    expect(screen.getByText('openai/whisper-large-v3-turbo')).toBeTruthy();
    expect(screen.getByText('hexgrad/Kokoro-82M')).toBeTruthy();
  });

  it('renders Live badge and toggle', () => {
    render(
      <I18nProvider>
        <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
      </I18nProvider>
    );

    expect(screen.getAllByText(/Gemma 4 E4B/i).length).toBeGreaterThan(0);

    const toggleBtn = screen.getByRole('button', { name: /Mode|Modalità/i });
    expect(toggleBtn).toBeTruthy();
    fireEvent.click(toggleBtn);
    expect(screen.getByText(/Standard Mode|Modalità Standard/i)).toBeTruthy();
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

  describe('Audio Resampling & Buffer Helpers', () => {
    it('passes through 16kHz PCM data without resampling', () => {
      const input = new Float32Array([0.0, 0.5, -0.5, 1.0, -1.0]);
      const resampled = downsampleTo16k(input, 16000);
      expect(resampled.length).toBe(5);
      expect(resampled[0]).toBe(0);
      expect(resampled[1]).toBe(Math.round(0.5 * 32767));
      expect(resampled[2]).toBe(-16384);
      expect(resampled[3]).toBe(32767);
      expect(resampled[4]).toBe(-32768);
    });

    it('downsamples 48kHz audio to 16000Hz (3x reduction) on mobile phone input', () => {
      // 48000Hz buffer of 480 samples -> should produce exactly 160 samples at 16kHz
      const input = new Float32Array(480);
      for (let i = 0; i < input.length; i++) {
        input[i] = 0.5;
      }
      const resampled = downsampleTo16k(input, 48000);
      expect(resampled.length).toBe(160);
      expect(resampled[0]).toBe(Math.round(0.5 * 32767));
      expect(resampled[159]).toBe(Math.round(0.5 * 32767));
    });

    it('downsamples 44.1kHz audio to 16000Hz accurately', () => {
      // 441 samples at 44.1kHz = 10ms -> 160 samples at 16kHz
      const input = new Float32Array(441);
      for (let i = 0; i < input.length; i++) {
        input[i] = 0.25;
      }
      const resampled = downsampleTo16k(input, 44100);
      expect(resampled.length).toBe(160);
      expect(resampled[50]).toBe(Math.round(0.25 * 32767));
    });

    it('clamps audio samples within Int16 limits (-32768 to 32767)', () => {
      const input = new Float32Array([1.5, -2.0, 0.0]);
      const resampled = downsampleTo16k(input, 16000);
      expect(resampled[0]).toBe(32767);
      expect(resampled[1]).toBe(-32768);
      expect(resampled[2]).toBe(0);
    });

    it('converts base64 string to ArrayBuffer correctly', () => {
      // 'QUJD' is base64 for 'ABC' (0x41, 0x42, 0x43)
      const buffer = base64ToArrayBuffer('QUJD');
      expect(buffer.byteLength).toBe(3);
      const uint8 = new Uint8Array(buffer);
      expect(uint8[0]).toBe(0x41);
      expect(uint8[1]).toBe(0x42);
      expect(uint8[2]).toBe(0x43);
    });

    it('decodes audio buffer with Promise-based decodeAudioData', async () => {
      const mockAudioBuffer = { duration: 1.5, sampleRate: 16000 };
      const mockAudioCtx = {
        decodeAudioData: vi.fn().mockResolvedValue(mockAudioBuffer)
      };
      const arrBuf = new ArrayBuffer(8);
      const res = await decodeAudioBuffer(mockAudioCtx, arrBuf);
      expect(res).toBe(mockAudioBuffer);
    });

    it('decodes audio buffer with callback-based decodeAudioData', async () => {
      const mockAudioBuffer = { duration: 2.0, sampleRate: 48000 };
      const mockAudioCtx = {
        decodeAudioData: vi.fn((buf, success) => {
          success(mockAudioBuffer);
        })
      };
      const arrBuf = new ArrayBuffer(8);
      const res = await decodeAudioBuffer(mockAudioCtx, arrBuf);
      expect(res).toBe(mockAudioBuffer);
    });
  });
});
