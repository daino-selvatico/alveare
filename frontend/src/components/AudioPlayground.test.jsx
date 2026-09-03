import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import AudioPlayground, { GaplessPCMPlayer } from './AudioPlayground';
import { I18nProvider } from '../i18n/I18nContext';

// Mock Web Audio API for GaplessPCMPlayer unit tests
class MockAudioContext {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 44100;
    this.state = 'running';
    this.currentTime = 0.5;
    this.destination = {};
  }
  createBuffer(channels, length, sampleRate) {
    return {
      duration: length / sampleRate,
      length,
      sampleRate,
      numberOfChannels: channels,
      copyToChannel: vi.fn()
    };
  }
  createBufferSource() {
    return {
      buffer: null,
      connect: vi.fn(),
      disconnect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      onended: null
    };
  }
  resume() {
    this.state = 'running';
    return Promise.resolve();
  }
  close() {
    this.state = 'closed';
    return Promise.resolve();
  }
}

describe('GaplessPCMPlayer', () => {
  beforeEach(() => {
    window.AudioContext = MockAudioContext;
  });

  it('initializes with default prebuffer_chunks = 2', () => {
    const player = new GaplessPCMPlayer(44100);
    expect(player.prebufferChunks).toBe(2);
    expect(player.pendingChunks).toEqual([]);
    expect(player.isPlaying).toBe(false);
  });

  it('accumulates chunks in pendingChunks when prebufferChunks = 2 until threshold is reached', () => {
    const player = new GaplessPCMPlayer(44100, 2);
    const scheduled = [];
    player.onChunkScheduled = (meta, startTime, dur) => {
      scheduled.push({ meta, startTime, dur });
    };

    // Create 16-bit PCM dummy data (1000 samples = 2000 bytes)
    const chunk1 = new Int16Array(1000).buffer;
    const chunk2 = new Int16Array(1000).buffer;

    // Enqueue chunk 1: should be buffered in pendingChunks, not yet playing
    const res1 = player.enqueuePCM16(chunk1, { text: 'Prima frase' });
    expect(res1).toBe(0);
    expect(player.pendingChunks.length).toBe(1);
    expect(player.isPlaying).toBe(false);
    expect(scheduled.length).toBe(0);

    // Enqueue chunk 2: threshold 2 reached -> flushes both chunks and starts playback
    const res2 = player.enqueuePCM16(chunk2, { text: 'Seconda frase' });
    expect(res2).toBeGreaterThan(0);
    expect(player.pendingChunks.length).toBe(0);
    expect(player.isPlaying).toBe(true);
    expect(scheduled.length).toBe(2);
    expect(scheduled[0].meta.text).toBe('Prima frase');
    expect(scheduled[1].meta.text).toBe('Seconda frase');
  });

  it('flushes pending prebuffer immediately when stream completes early with fewer chunks', () => {
    const player = new GaplessPCMPlayer(44100, 3);
    const scheduled = [];
    player.onChunkScheduled = (meta) => {
      scheduled.push(meta.text);
    };

    const chunk1 = new Int16Array(500).buffer;
    player.enqueuePCM16(chunk1, { text: 'Frase singola' });

    expect(player.pendingChunks.length).toBe(1);
    expect(player.isPlaying).toBe(false);

    // Server sends completed event -> client triggers flushPendingPrebuffer()
    player.flushPendingPrebuffer();

    expect(player.pendingChunks.length).toBe(0);
    expect(player.isPlaying).toBe(true);
    expect(scheduled).toEqual(['Frase singola']);
  });

  it('schedules immediately when prebufferChunks = 1 (zero latency)', () => {
    const player = new GaplessPCMPlayer(44100, 1);
    const chunk1 = new Int16Array(500).buffer;
    const startTime = player.enqueuePCM16(chunk1, { text: 'Zero latency chunk' });

    expect(startTime).toBeGreaterThan(0);
    expect(player.pendingChunks.length).toBe(0);
    expect(player.isPlaying).toBe(true);
  });

  it('clears pending prebuffer and active nodes on stopAndFlush', () => {
    const player = new GaplessPCMPlayer(44100, 3);
    const chunk1 = new Int16Array(500).buffer;
    player.enqueuePCM16(chunk1, { text: 'Buffered chunk' });
    expect(player.pendingChunks.length).toBe(1);

    player.stopAndFlush();
    expect(player.pendingChunks.length).toBe(0);
    expect(player.isPlaying).toBe(false);
  });
});

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

  it('switches to TTS mode and displays active hardware device badge', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={{ ...mockStatus, model: 'audio8-0.1b', device: 'npu' }}
          activeModel="audio8-0.1b"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    const ttsTabBtn = screen.getByText(/Text to Speech|Sintesi Vocale/i);
    fireEvent.click(ttsTabBtn);

    const deviceBadge = screen.getByText(/⚡ NPU/i);
    expect(deviceBadge).toBeDefined();
    expect(screen.getByText(/audio8-0.1b/i)).toBeDefined();
  });

  it('enables streaming mode and displays live teleprompter with pre-buffering slider', () => {
    render(
      <I18nProvider>
        <AudioPlayground
          apiBase={mockApiBase}
          status={{ ...mockStatus, model: 'audio8-0.1b', device: 'npu' }}
          activeModel="audio8-0.1b"
          isServerRunning={true}
        />
      </I18nProvider>
    );

    const ttsTabBtn = screen.getByText(/Text to Speech|Sintesi Vocale/i);
    fireEvent.click(ttsTabBtn);

    const streamingToggleBtn = screen.getByText(/Modalità Streaming|Streaming/i);
    fireEvent.click(streamingToggleBtn);

    expect(screen.getByText(/SOTTOTITOLI LIVE AUDIO|LIVE AUDIO TELEPROMPTER/i)).toBeDefined();
    expect(screen.getByText(/Parametri Streaming Real-Time/i)).toBeDefined();
    expect(screen.getByText(/Pre-buffering di Avvio/i)).toBeDefined();
    expect(screen.getByText(/Sintetizza in Streaming/i)).toBeDefined();

    const prebufferSlider = screen.getByLabelText(/Pre-buffering di Avvio/i);
    expect(prebufferSlider).toBeDefined();
    expect(prebufferSlider.value).toBe('2');

    fireEvent.change(prebufferSlider, { target: { value: '4' } });
    expect(prebufferSlider.value).toBe('4');
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

