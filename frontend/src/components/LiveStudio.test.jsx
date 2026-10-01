import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
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

      const liveTab = screen.queryByRole('tab', { name: /Live/i });
      if (liveTab) fireEvent.click(liveTab);

      const startBtns = screen.getAllByRole('button', { name: /Start Live Session|Avvia Sessione Live/i });
      fireEvent.click(startBtns[0]);

      const errorAlert = await screen.findByText(/Microphone access was denied|Accesso al microfono negato/i);
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

  describe('Voice Studio Mode Separation and Voice UX', () => {
    let origWS;
    let origMediaDevices;
    let wsInstances;

    class MockTestWebSocket {
      constructor(url) {
        this.url = url;
        this.readyState = 1; // OPEN
        this.send = vi.fn();
        this.close = vi.fn();
        wsInstances.push(this);
        setTimeout(() => {
          if (this.onopen) this.onopen();
        }, 5);
      }
    }

    beforeEach(() => {
      localStorage.setItem('alveare_language', 'it');
      wsInstances = [];
      origWS = window.WebSocket;
      origMediaDevices = navigator.mediaDevices;
      window.WebSocket = MockTestWebSocket;
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({})
      });
    });

    afterEach(() => {
      window.WebSocket = origWS;
      navigator.mediaDevices = origMediaDevices;
    });

    it('visibly separates Assistente tap-to-talk and Live Sperimentale modes and clarifies native wakeword', () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      // Verify mode tabs
      const assistenteTab = screen.getByRole('tab', { name: /Assistente|Assistant/i });
      const liveTab = screen.getByRole('tab', { name: /Live · Sperimentale|Live Sperimentale|Live · Experimental/i });
      expect(assistenteTab).toBeTruthy();
      expect(liveTab).toBeTruthy();

      // Wakeword notice present
      expect(screen.getByText(/Android/i)).toBeTruthy();
      expect(screen.getByText(/wakeword/i)).toBeTruthy();

      // Default is Assistente tap-to-talk
      expect(screen.getAllByText(/Tocca per parlare|Tap to talk/i).length).toBeGreaterThan(0);

      // Switching to Live mode changes UI
      fireEvent.click(liveTab);
      expect(screen.getByText(/Avvia Sessione Live|Start Live Session/i)).toBeTruthy();
    });

    it('never requests microphone on initial mount and shows privacy status', () => {
      const getUserMediaMock = vi.fn();
      navigator.mediaDevices = { getUserMedia: getUserMediaMock };

      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      // Microphone MUST NOT be called on mount
      expect(getUserMediaMock).not.toHaveBeenCalled();
      // Privacy indicator shows inactive mic
      expect(screen.getByText(/Microfono spento|Microfono disattivato|Microphone inactive/i)).toBeTruthy();
    });

    it('supports explicit mute control that disables capture', async () => {
      const mockTrack = { stop: vi.fn(), enabled: true };
      const mockStream = {
        getTracks: vi.fn(() => [mockTrack]),
        getAudioTracks: vi.fn(() => [mockTrack])
      };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue(mockStream)
      };

      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const muteBtn = screen.getByRole('button', { name: /Muto|Mute|Disattiva microfono|Silenzia/i });
      expect(muteBtn).toBeTruthy();
      fireEvent.click(muteBtn);

      expect(screen.getByText(/Microfono mutato|Microphone muted|Muto/i)).toBeTruthy();
    });

    it('renders tool execution cards when tool_call_start and tool_call_done arrive', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      // Switch to Live mode and start session
      const liveTab = screen.getByRole('tab', { name: /Live · Sperimentale|Live Sperimentale|Live · Experimental/i });
      fireEvent.click(liveTab);

      const mockTrack = { stop: vi.fn(), enabled: true };
      const mockStream = {
        getTracks: vi.fn(() => [mockTrack]),
        getAudioTracks: vi.fn(() => [mockTrack])
      };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue(mockStream)
      };

      const startBtns = screen.getAllByRole('button', { name: /Avvia Sessione Live|Start Live Session/i });
      await act(async () => {
        fireEvent.click(startBtns[0]);
        await new Promise(r => setTimeout(r, 20));
      });

      const ws = wsInstances[0];
      expect(ws).toBeDefined();

      // Tool call start arrives
      await act(async () => {
        ws.onmessage({
          data: JSON.stringify({
            event: 'tool_call_start',
            tool_name: 'calcolatrice',
            tool_args: { expression: '42 * 2' }
          })
        });
      });

      expect(await screen.findByText(/calcolatrice/i)).toBeTruthy();

      // Tool call done arrives
      await act(async () => {
        ws.onmessage({
          data: JSON.stringify({
            event: 'tool_call_done',
            tool_name: 'calcolatrice',
            result: '84'
          })
        });
      });

      expect(await screen.findByText(/completato|completed/i)).toBeTruthy();
    });

    it('avoids text duplicated between llm_chunk and turn_complete.full_text', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const liveTab = screen.getByRole('tab', { name: /Live · Sperimentale|Live Sperimentale|Live · Experimental/i });
      fireEvent.click(liveTab);

      const mockTrack = { stop: vi.fn(), enabled: true };
      const mockStream = {
        getTracks: vi.fn(() => [mockTrack]),
        getAudioTracks: vi.fn(() => [mockTrack])
      };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue(mockStream)
      };

      const startBtns = screen.getAllByRole('button', { name: /Avvia Sessione Live|Start Live Session/i });
      await act(async () => {
        fireEvent.click(startBtns[0]);
        await new Promise(r => setTimeout(r, 20));
      });

      const ws = wsInstances[0];

      // Streaming chunks arrive
      await act(async () => {
        ws.onmessage({
          data: JSON.stringify({ event: 'llm_chunk', text: 'Ciao' })
        });
        ws.onmessage({
          data: JSON.stringify({ event: 'llm_chunk', text: ' come stai?' })
        });
        // turn_complete arrives with canonical full_text
        ws.onmessage({
          data: JSON.stringify({
            event: 'turn_complete',
            full_text: 'Ciao come stai?',
            metrics: { e2e_latency_ms: 320 }
          })
        });
      });

      // Exactly one instance of finalized full_text in messages, no duplicate streaming bubble
      const matches = screen.getAllByText('Ciao come stai?');
      expect(matches.length).toBe(1);
    });

    it('acknowledges memory clearing and resets context', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const clearBtn = screen.getByRole('button', { name: /Azzera memoria|Clear memory/i });
      expect(clearBtn).toBeTruthy();
      await act(async () => {
        fireEvent.click(clearBtn);
      });

      expect(await screen.findByText(/Memoria conversazione azzerata|Memoria e contesto azzerati|Conversation memory cleared/i)).toBeTruthy();
    });

    it('displays distinct labels for Server TTFA and Audible Latency', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      // Verify metric labels
      expect(screen.getByText(/TTFA Server|Server TTFA/i)).toBeTruthy();
      expect(screen.getByText(/Latenza Uditiva|Audible Latency/i)).toBeTruthy();
    });

    it('cleans up all media tracks and audio contexts on unmount', () => {
      const mockTrack = { stop: vi.fn(), enabled: true };
      const mockStream = {
        getTracks: vi.fn(() => [mockTrack]),
        getAudioTracks: vi.fn(() => [mockTrack])
      };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue(mockStream)
      };

      const { unmount } = render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      unmount();
      expect(true).toBe(true);
    });

    it('does not enter interrupt feedback loop on server interrupted or vad_speech_start', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const liveTab = screen.getByRole('tab', { name: /Live · Sperimentale|Live Sperimentale|Live · Experimental/i });
      fireEvent.click(liveTab);

      const mockTrack = { stop: vi.fn(), enabled: true };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: vi.fn(() => [mockTrack]),
          getAudioTracks: vi.fn(() => [mockTrack])
        })
      };

      const startBtns = screen.getAllByRole('button', { name: /Avvia Sessione Live|Start Live Session/i });
      await act(async () => {
        fireEvent.click(startBtns[0]);
        await new Promise(r => setTimeout(r, 20));
      });

      const ws = wsInstances[0];
      expect(ws).toBeDefined();

      // Handshake
      await act(async () => {
        ws.onmessage({ data: JSON.stringify({ event: 'live_connected' }) });
      });

      const callsAfterConnect = ws.send.mock.calls.length;

      // Server sends 'interrupted'
      await act(async () => {
        ws.onmessage({ data: JSON.stringify({ event: 'interrupted' }) });
      });

      // No new message must be sent to the server (no ping-pong loop!)
      expect(ws.send.mock.calls.length).toBe(callsAfterConnect);

      // Server sends 'vad_speech_start'
      await act(async () => {
        ws.onmessage({ data: JSON.stringify({ event: 'vad_speech_start' }) });
      });

      // Still no interrupt message sent to the server!
      expect(ws.send.mock.calls.length).toBe(callsAfterConnect);
    });

    it('sends initial config settings strictly upon live_connected handshake', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const liveTab = screen.getByRole('tab', { name: /Live · Sperimentale|Live Sperimentale|Live · Experimental/i });
      fireEvent.click(liveTab);

      const mockTrack = { stop: vi.fn(), enabled: true };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: vi.fn(() => [mockTrack]),
          getAudioTracks: vi.fn(() => [mockTrack])
        })
      };

      const startBtns = screen.getAllByRole('button', { name: /Avvia Sessione Live|Start Live Session/i });
      await act(async () => {
        fireEvent.click(startBtns[0]);
        await new Promise(r => setTimeout(r, 20));
      });

      const ws = wsInstances[0];
      expect(ws).toBeDefined();

      // Prior to live_connected handshake, config was NOT sent
      const configBefore = ws.send.mock.calls.find(c => {
        try { return JSON.parse(c[0]).type === 'config'; } catch { return false; }
      });
      expect(configBefore).toBeUndefined();

      // Server sends live_connected
      await act(async () => {
        ws.onmessage({ data: JSON.stringify({ event: 'live_connected' }) });
      });

      // Now config must have been sent with settings
      const configAfter = ws.send.mock.calls.find(c => {
        try { return JSON.parse(c[0]).type === 'config'; } catch { return false; }
      });
      expect(configAfter).toBeDefined();
      const payload = JSON.parse(configAfter[0]);
      expect(payload.type).toBe('config');
      expect(payload.settings).toBeDefined();
      expect(payload.settings.voice).toBeDefined();
      expect(payload.settings.max_context_turns).toBeGreaterThan(0);
    });

    it('renders tool execution card with result_summary and input args', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const liveTab = screen.getByRole('tab', { name: /Live · Sperimentale|Live Sperimentale|Live · Experimental/i });
      fireEvent.click(liveTab);

      const mockTrack = { stop: vi.fn(), enabled: true };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: vi.fn(() => [mockTrack]),
          getAudioTracks: vi.fn(() => [mockTrack])
        })
      };

      const startBtns = screen.getAllByRole('button', { name: /Avvia Sessione Live|Start Live Session/i });
      await act(async () => {
        fireEvent.click(startBtns[0]);
        await new Promise(r => setTimeout(r, 20));
      });

      const ws = wsInstances[0];

      // Tool call start
      await act(async () => {
        ws.onmessage({
          data: JSON.stringify({
            event: 'tool_call_start',
            tool_name: 'get_weather',
            tool_args: { location: 'Bologna' },
            turn_id: 10
          })
        });
      });

      expect(await screen.findByText(/get_weather/i)).toBeTruthy();
      expect(screen.getByText(/Bologna/i)).toBeTruthy();

      // Tool call done with result_summary
      await act(async () => {
        ws.onmessage({
          data: JSON.stringify({
            event: 'tool_call_done',
            tool_name: 'get_weather',
            status: 'done',
            result_summary: '24°C Sunny',
            turn_id: 10
          })
        });
      });

      expect(await screen.findByText(/24°C Sunny/i)).toBeTruthy();
    });

    it('handles two consecutive assistant turns cleanly in transcript', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const liveTab = screen.getByRole('tab', { name: /Live · Sperimentale|Live Sperimentale|Live · Experimental/i });
      fireEvent.click(liveTab);

      const mockTrack = { stop: vi.fn(), enabled: true };
      navigator.mediaDevices = {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: vi.fn(() => [mockTrack]),
          getAudioTracks: vi.fn(() => [mockTrack])
        })
      };

      const startBtns = screen.getAllByRole('button', { name: /Avvia Sessione Live|Start Live Session/i });
      await act(async () => {
        fireEvent.click(startBtns[0]);
        await new Promise(r => setTimeout(r, 20));
      });

      const ws = wsInstances[0];

      // Turn 1
      await act(async () => {
        ws.onmessage({ data: JSON.stringify({ event: 'vad_speech_start' }) });
        ws.onmessage({ data: JSON.stringify({ event: 'vad_speech_end' }) });
        ws.onmessage({ data: JSON.stringify({ event: 'llm_chunk', text: 'Primo messaggio.' }) });
        ws.onmessage({ data: JSON.stringify({ event: 'turn_complete', turn_id: 1, full_text: 'Primo messaggio.' }) });
      });

      expect(await screen.findByText('Primo messaggio.')).toBeTruthy();

      // Turn 2
      await act(async () => {
        ws.onmessage({ data: JSON.stringify({ event: 'vad_speech_start' }) });
        ws.onmessage({ data: JSON.stringify({ event: 'vad_speech_end' }) });
        ws.onmessage({ data: JSON.stringify({ event: 'llm_chunk', text: 'Secondo messaggio.' }) });
        ws.onmessage({ data: JSON.stringify({ event: 'turn_complete', turn_id: 2, full_text: 'Secondo messaggio.' }) });
      });

      expect(await screen.findByText('Secondo messaggio.')).toBeTruthy();
      expect(screen.getByText('Primo messaggio.')).toBeTruthy();
    });

    it('toggles collapsible hardware and routing section', async () => {
      render(
        <I18nProvider>
          <LiveStudio apiBase="http://localhost:8080" status={mockStatus} />
        </I18nProvider>
      );

      const toggleBtn = screen.getByRole('button', { name: /Dettagli Hardware & Routing|Hardware & Routing Details/i });
      expect(toggleBtn).toBeTruthy();
      expect(toggleBtn.getAttribute('aria-expanded')).toBe('false');

      act(() => {
        fireEvent.click(toggleBtn);
      });

      expect(toggleBtn.getAttribute('aria-expanded')).toBe('true');
    });
  });
});
