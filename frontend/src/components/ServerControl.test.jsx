import React from 'react';
import { render, screen, fireEvent, waitFor, within, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import ServerControl, { deduplicateModelOptions } from './ServerControl';
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

  it('deduplicates model options and normalizes to canonical IDs', () => {
    // STT duplicate input
    const sttRaw = ['whisper-base', 'openai/whisper-base', 'whisper-large-v3-turbo'];
    const sttDedup = deduplicateModelOptions(sttRaw);
    expect(sttDedup).toHaveLength(2);
    expect(sttDedup[0]).toEqual({ id: 'openai/whisper-base', label: 'Whisper Base' });
    expect(sttDedup[1]).toEqual({ id: 'openai/whisper-large-v3-turbo', label: 'Whisper Large v3 Turbo' });

    // TTS duplicate input
    const ttsRaw = ['audio8-0.1b', 'Audio8/Audio8-TTS-Preview-0.1b', 'Audio8/Audio8-TTS-Preview-0.6b'];
    const ttsDedup = deduplicateModelOptions(ttsRaw);
    expect(ttsDedup).toHaveLength(2);
    expect(ttsDedup[0]).toEqual({ id: 'Audio8/Audio8-TTS-Preview-0.1b', label: 'Audio8 0.1B' });
    expect(ttsDedup[1]).toEqual({ id: 'Audio8/Audio8-TTS-Preview-0.6b', label: 'Audio8 0.6B' });
  });

  it('renders user-friendly labels with canonical value IDs in dropdown selects without duplicates', () => {
    const modelsWithDuplicates = [
      { id: 'whisper-base', name: 'whisper-base', task: 'speech-to-text' },
      { id: 'openai/whisper-base', name: 'Whisper Base', task: 'speech-to-text' },
      { id: 'audio8-0.1b', name: 'audio8-0.1b', task: 'text-to-speech' },
      { id: 'Audio8/Audio8-TTS-Preview-0.1b', name: 'Audio8 TTS 0.1B', task: 'text-to-speech' },
      { id: 'gemma4', name: 'Gemma 4 (12B)', task: 'text-generation' }
    ];

    const { container } = render(
      <I18nProvider>
        <ServerControl
          apiBase="http://localhost:8080"
          status={mockStatus}
          models={modelsWithDuplicates}
          modelsLoading={false}
          onRefresh={vi.fn()}
        />
      </I18nProvider>
    );

    const selects = container.querySelectorAll('select');
    expect(selects.length).toBeGreaterThanOrEqual(3);

    // LLM select options
    const llmSelect = selects[0];
    const llmOptions = Array.from(llmSelect.querySelectorAll('option'));
    const llmValues = llmOptions.map(o => o.value);
    expect(new Set(llmValues).size).toBe(llmValues.length); // no duplicates
    expect(llmOptions.some(o => o.textContent === 'Gemma 4 12B')).toBe(true);

    // STT select options
    const sttSelect = selects[1];
    const sttOptions = Array.from(sttSelect.querySelectorAll('option'));
    const sttValues = sttOptions.map(o => o.value);
    expect(new Set(sttValues).size).toBe(sttValues.length); // no duplicates
    const whisperBaseOpt = sttOptions.find(o => o.value === 'openai/whisper-base');
    expect(whisperBaseOpt).toBeDefined();
    expect(whisperBaseOpt.textContent).toBe('Whisper Base');
    // Ensure duplicate 'whisper-base' is not an option value
    expect(sttValues.filter(v => v === 'whisper-base')).toHaveLength(0);

    // TTS select options
    const ttsSelect = selects[2];
    const ttsOptions = Array.from(ttsSelect.querySelectorAll('option'));
    const ttsValues = ttsOptions.map(o => o.value);
    expect(new Set(ttsValues).size).toBe(ttsValues.length); // no duplicates
    const audio8Opt = ttsOptions.find(o => o.value === 'Audio8/Audio8-TTS-Preview-0.1b');
    expect(audio8Opt).toBeDefined();
    expect(audio8Opt.textContent).toBe('Audio8 0.1B');
    // Ensure duplicate 'audio8-0.1b' is not an option value
    expect(ttsValues.filter(v => v === 'audio8-0.1b')).toHaveLength(0);
  });

  it('correctly maps and deduplicates Kokoro 82M TTS model options', () => {
    const kokoroRaw = ['kokoro', 'kokoro-82m', 'hexgrad/Kokoro-82M', 'cocore'];
    const kokoroDedup = deduplicateModelOptions(kokoroRaw);
    expect(kokoroDedup).toHaveLength(1);
    expect(kokoroDedup[0]).toEqual({ id: 'hexgrad/Kokoro-82M', label: 'Kokoro 82M' });
  });

  it('opens and dismisses buildTargetModal using the X button', () => {
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

    const compileBtn = screen.getByLabelText(`Compila kernel per ${mockModels[0].id}`);
    expect(compileBtn).toBeTruthy();
    fireEvent.click(compileBtn);

    // Modal is open
    const modal = screen.getByRole('dialog');
    expect(modal).toBeTruthy();
    expect(within(modal).getByText(mockModels[0].id)).toBeTruthy();

    // Close using the X button
    const closeBtn = within(modal).getByRole('button', { name: '' });
    expect(closeBtn).toBeTruthy();
    act(() => {
      fireEvent.click(closeBtn);
    });

    // Modal is dismissed
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});
