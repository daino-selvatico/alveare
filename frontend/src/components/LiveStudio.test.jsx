import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
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
});
