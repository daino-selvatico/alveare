import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AudioPlaybackQueue } from './audioQueue';

describe('AudioPlaybackQueue', () => {
  let mockAudioContext;
  let mockSource;

  beforeEach(() => {
    mockSource = {
      buffer: null,
      connect: vi.fn(),
      disconnect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      onended: null
    };

    mockAudioContext = {
      state: 'running',
      resume: vi.fn().mockResolvedValue(),
      createBufferSource: vi.fn(() => ({ ...mockSource })),
      destination: {},
      decodeAudioData: vi.fn((buffer, success) => {
        const mockBuffer = { duration: 1.0, sampleRate: 16000 };
        if (success) success(mockBuffer);
        return Promise.resolve(mockBuffer);
      })
    };
  });

  it('keeps turn active when turn_complete arrives before audio playback finishes', async () => {
    const onSpeakingChange = vi.fn();
    const onTurnEnd = vi.fn();

    const queue = new AudioPlaybackQueue({
      getAudioContext: () => mockAudioContext,
      onSpeakingChange,
      onTurnEnd
    });

    // Enqueue a chunk
    await queue.enqueue({ audioB64: 'QUJD', turnId: 1 });
    expect(onSpeakingChange).toHaveBeenCalledWith(true);

    // Backend sends turn_complete while audio is still playing
    queue.markTurnComplete(1);

    // onTurnEnd should NOT have fired yet because audio source hasn't ended
    expect(onTurnEnd).not.toHaveBeenCalled();
    expect(queue.isSpeaking()).toBe(true);

    // Audio source finishes playing
    queue.simulateCurrentSourceEnd();

    // Now onTurnEnd should have fired
    expect(onTurnEnd).toHaveBeenCalledWith(1);
    expect(onSpeakingChange).toHaveBeenCalledWith(false);
  });

  it('discards stale in-flight decode when interrupted', async () => {
    let delayedResolve;
    mockAudioContext.decodeAudioData = vi.fn(() => new Promise((resolve) => {
      delayedResolve = resolve;
    }));

    const onSpeakingChange = vi.fn();
    const queue = new AudioPlaybackQueue({
      getAudioContext: () => mockAudioContext,
      onSpeakingChange
    });

    // Start enqueuing
    const enqueuePromise = queue.enqueue({ audioB64: 'QUJD', turnId: 1 });

    // User interrupts before decode resolves
    queue.interrupt();
    expect(onSpeakingChange).toHaveBeenCalledWith(false);

    // Now decode resolves late
    if (delayedResolve) {
      delayedResolve({ duration: 1.0, sampleRate: 16000 });
    }
    await enqueuePromise;

    // createBufferSource should NOT be started for the stale decode
    expect(mockAudioContext.createBufferSource).not.toHaveBeenCalled();
    expect(queue.isSpeaking()).toBe(false);
  });

  it('bounds the audio queue to maxQueueSize and drops oldest on overflow', () => {
    // Prevent immediate playback so items stay queued
    const queue = new AudioPlaybackQueue({
      getAudioContext: () => mockAudioContext,
      maxQueueSize: 3
    });
    queue.pausePlaybackWorker(); // hold queue

    queue.enqueueChunkData({ audioB64: 'chunk1', turnId: 1 });
    queue.enqueueChunkData({ audioB64: 'chunk2', turnId: 1 });
    queue.enqueueChunkData({ audioB64: 'chunk3', turnId: 1 });
    expect(queue.getQueueLength()).toBe(3);

    // Enqueue 4th item -> should drop chunk1
    queue.enqueueChunkData({ audioB64: 'chunk4', turnId: 1 });
    expect(queue.getQueueLength()).toBe(3);
    const first = queue.peekFirstChunk();
    expect(first.audioB64).toBe('chunk2');
  });

  it('reports audible latency separately from server TTFA', async () => {
    const onAudibleLatency = vi.fn();
    const queue = new AudioPlaybackQueue({
      getAudioContext: () => mockAudioContext,
      onAudibleLatency
    });

    queue.setTurnStartTime(1000);
    // Mock Date.now during playback start
    const dateSpy = vi.spyOn(Date, 'now').mockReturnValue(1450);

    await queue.enqueue({ audioB64: 'QUJD', turnId: 1 });

    expect(onAudibleLatency).toHaveBeenCalledWith(450);
    dateSpy.mockRestore();
  });

  it('supports client TTS mode when configured and available', async () => {
    const mockSpeak = vi.fn();
    const origSpeechSynthesis = window.speechSynthesis;
    const origUtterance = window.SpeechSynthesisUtterance;
    window.speechSynthesis = {
      speak: mockSpeak,
      cancel: vi.fn()
    };
    window.SpeechSynthesisUtterance = class MockUtterance {
      constructor(text) {
        this.text = text;
        this.onend = null;
      }
    };

    try {
      const onSpeakingChange = vi.fn();
      const onTurnEnd = vi.fn();

      const queue = new AudioPlaybackQueue({
        getAudioContext: () => mockAudioContext,
        ttsMode: 'client',
        onSpeakingChange,
        onTurnEnd
      });

      queue.speakClientText('Ciao come stai?', 1);
      expect(mockSpeak).toHaveBeenCalled();
      expect(onSpeakingChange).toHaveBeenCalledWith(true);
    } finally {
      window.speechSynthesis = origSpeechSynthesis;
      window.SpeechSynthesisUtterance = origUtterance;
    }
  });

  it('does not corrupt isPlaying when an old decode resolves while a new playback session is active', async () => {
    let oldResolve;
    mockAudioContext.decodeAudioData = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => {
        oldResolve = resolve;
      }))
      .mockImplementationOnce(() => Promise.resolve({ duration: 1.0, sampleRate: 16000 }));

    const onSpeakingChange = vi.fn();
    const queue = new AudioPlaybackQueue({
      getAudioContext: () => mockAudioContext,
      onSpeakingChange
    });

    // 1. Start generation 1 decode (in-flight)
    const turn1Promise = queue.enqueue({ audioB64: 'QUJD', turnId: 1 });

    // 2. User interrupts generation 1
    queue.interrupt();
    expect(queue.isPlaying).toBe(false);

    // 3. New turn starts (generation 2) and begins playing
    await queue.enqueue({ audioB64: 'QUJD', turnId: 2 });
    expect(queue.isPlaying).toBe(true);

    // 4. Old decode from generation 1 resolves now
    if (oldResolve) {
      oldResolve({ duration: 1.0, sampleRate: 16000 });
    }
    await turn1Promise;

    // 5. Stale generation MUST NOT mutate newer generation's isPlaying state
    expect(queue.isPlaying).toBe(true);
  });

  it('handles back-to-back turns with exactly one onTurnEnd per turn', async () => {
    const onTurnEnd = vi.fn();
    const queue = new AudioPlaybackQueue({
      getAudioContext: () => mockAudioContext,
      onTurnEnd
    });

    // Turn 1
    await queue.enqueue({ audioB64: 'QUJD', turnId: 1 });
    queue.markTurnComplete(1);
    expect(onTurnEnd).not.toHaveBeenCalled();
    queue.simulateCurrentSourceEnd();
    expect(onTurnEnd).toHaveBeenCalledTimes(1);
    expect(onTurnEnd).toHaveBeenLastCalledWith(1);

    // Turn 2
    await queue.enqueue({ audioB64: 'QUJD', turnId: 2 });
    queue.markTurnComplete(2);
    expect(onTurnEnd).toHaveBeenCalledTimes(1); // not called for turn 2 yet
    queue.simulateCurrentSourceEnd();
    expect(onTurnEnd).toHaveBeenCalledTimes(2);
    expect(onTurnEnd).toHaveBeenLastCalledWith(2);
  });

  it('multi-sentence native TTS does not signal turn end before all sentences finish', () => {
    const utterances = [];
    const mockSpeak = vi.fn((utt) => utterances.push(utt));
    const origSpeechSynthesis = window.speechSynthesis;
    const origUtterance = window.SpeechSynthesisUtterance;

    window.speechSynthesis = {
      speak: mockSpeak,
      cancel: vi.fn()
    };
    window.SpeechSynthesisUtterance = class MockUtterance {
      constructor(text) {
        this.text = text;
        this.onstart = null;
        this.onend = null;
        this.onerror = null;
      }
    };

    try {
      const onTurnEnd = vi.fn();
      const queue = new AudioPlaybackQueue({
        ttsMode: 'client',
        onTurnEnd
      });

      // Queue 2 sentences for turn 1
      queue.speakClientText('Prima frase.', 1);
      queue.speakClientText('Seconda frase.', 1);
      expect(utterances.length).toBe(2);

      // Backend marks turn complete while sentences are speaking
      queue.markTurnComplete(1);
      expect(onTurnEnd).not.toHaveBeenCalled();

      // First sentence finishes
      utterances[0].onend();
      expect(onTurnEnd).not.toHaveBeenCalled(); // Must NOT signal turn end early!

      // Second sentence finishes
      utterances[1].onend();
      expect(onTurnEnd).toHaveBeenCalledTimes(1);
      expect(onTurnEnd).toHaveBeenCalledWith(1);
    } finally {
      window.speechSynthesis = origSpeechSynthesis;
      window.SpeechSynthesisUtterance = origUtterance;
    }
  });

  it('does not stall the queue when decodeAudioData fails', async () => {
    mockAudioContext.decodeAudioData = vi.fn(() => Promise.reject(new Error('Corrupt audio stream')));
    const onTurnEnd = vi.fn();
    const queue = new AudioPlaybackQueue({
      getAudioContext: () => mockAudioContext,
      onTurnEnd
    });

    await queue.enqueue({ audioB64: 'QUJD', turnId: 42 });
    queue.markTurnComplete(42);

    // Queue must recover and signal turn end instead of hanging forever
    expect(onTurnEnd).toHaveBeenCalledWith(42);
    expect(queue.isPlaying).toBe(false);
  });
});
