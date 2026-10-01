/**
 * Audio playback queue for local voice assistant.
 * Ensures strictly ordered playback, bounded queue to prevent memory leaks,
 * decodes asynchronously without reordering, invalidates stale in-flight decodes,
 * and decouples generation turn_complete from audible playback completion.
 */

export function base64ToArrayBuffer(base64) {
  const binaryString = typeof window !== 'undefined'
    ? window.atob(base64)
    : Buffer.from(base64, 'base64').toString('binary');
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

export function decodeAudioBuffer(audioCtx, arrayBuffer) {
  return new Promise((resolve, reject) => {
    let resolved = false;
    const onOk = (buf) => {
      if (!resolved) {
        resolved = true;
        resolve(buf);
      }
    };
    const onErr = (err) => {
      if (!resolved) {
        resolved = true;
        reject(err);
      }
    };
    try {
      const res = audioCtx.decodeAudioData(arrayBuffer, onOk, onErr);
      if (res && typeof res.then === 'function') {
        res.then(onOk, onErr);
      }
    } catch (e) {
      onErr(e);
    }
  });
}

export class AudioPlaybackQueue {
  constructor({
    getAudioContext,
    ttsMode = 'server',
    onSpeakingChange = null,
    onTurnEnd = null,
    onAudibleLatency = null,
    onOverflow = null,
    maxQueueSize = 50
  } = {}) {
    this.getAudioContext = getAudioContext;
    this.ttsMode = ttsMode;
    this.onSpeakingChange = onSpeakingChange;
    this.onTurnEnd = onTurnEnd;
    this.onAudibleLatency = onAudibleLatency;
    this.onOverflow = onOverflow;
    this.maxQueueSize = maxQueueSize;

    this.queue = [];
    this.sessionToken = 0;
    this.currentSource = null;
    this.isPlaying = false;
    this.currentTurnId = null;
    this.turnCompleteReceived = false;
    this.turnEndedReported = false;
    this.workerPaused = false;
    this.turnStartTime = 0;
    this.audibleReported = false;
    this.overflowCount = 0;
    this.pendingUtterancesCount = 0;
  }

  setTtsMode(mode) {
    this.ttsMode = mode;
  }

  setTurnStartTime(time) {
    this.turnStartTime = time;
    this.audibleReported = false;
  }

  pausePlaybackWorker() {
    this.workerPaused = true;
  }

  resumePlaybackWorker() {
    this.workerPaused = false;
    this.processQueue();
  }

  getQueueLength() {
    return this.queue.length;
  }

  peekFirstChunk() {
    return this.queue[0];
  }

  enqueueChunkData(chunk) {
    if (chunk.turnId !== undefined && chunk.turnId !== this.currentTurnId) {
      this.currentTurnId = chunk.turnId;
      this.turnCompleteReceived = false;
      this.turnEndedReported = false;
    } else if (this.turnCompleteReceived && this.turnEndedReported) {
      this.turnCompleteReceived = false;
      this.turnEndedReported = false;
    }

    if (this.queue.length >= this.maxQueueSize) {
      this.overflowCount++;
      const dropped = this.queue[0];
      console.warn(`[AudioPlaybackQueue] Queue overflow (${this.maxQueueSize} chunks reached). Dropping chunk to protect memory.`);
      if (this.onOverflow) {
        this.onOverflow({
          queueLength: this.queue.length,
          maxQueueSize: this.maxQueueSize,
          overflowCount: this.overflowCount,
          dropped
        });
      }
      this.queue.shift();
    }
    this.queue.push(chunk);
    if (chunk.turnId !== undefined) {
      this.currentTurnId = chunk.turnId;
    }
  }

  async enqueue({ audioB64, turnId, sampleRate, format }) {
    this.enqueueChunkData({ audioB64, turnId, sampleRate, format });
    if (!this.isPlaying && !this.workerPaused) {
      await this.processQueue();
    }
  }

  _signalTurnEnd(turnId) {
    if (this.turnEndedReported) return;
    this.turnEndedReported = true;
    if (this.onSpeakingChange) {
      this.onSpeakingChange(false);
    }
    if (this.onTurnEnd) {
      this.onTurnEnd(turnId ?? this.currentTurnId);
    }
  }

  markTurnComplete(turnId) {
    this.turnCompleteReceived = true;
    if (turnId !== undefined) {
      this.currentTurnId = turnId;
    }

    if (!this.isPlaying && this.queue.length === 0 && this.pendingUtterancesCount === 0) {
      this._signalTurnEnd(this.currentTurnId);
    }
  }

  async processQueue() {
    if (this.workerPaused || this.isPlaying || this.queue.length === 0) {
      return;
    }

    const chunk = this.queue.shift();
    if (!chunk || !chunk.audioB64) {
      return;
    }

    this.isPlaying = true;
    if (this.onSpeakingChange) {
      this.onSpeakingChange(true);
    }

    const token = this.sessionToken;
    const audioCtx = this.getAudioContext ? this.getAudioContext() : null;

    if (!audioCtx) {
      if (this.sessionToken === token) {
        this.isPlaying = false;
        this.onChunkEnded(chunk.turnId);
      }
      return;
    }

    try {
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume().catch(() => {});
      }

      const arrayBuffer = base64ToArrayBuffer(chunk.audioB64);
      const audioBuffer = await decodeAudioBuffer(audioCtx, arrayBuffer);

      // Verify token hasn't changed during async decode.
      // Only the current generation owns the state: do NOT mutate isPlaying for stale decodes.
      if (this.sessionToken !== token) {
        return;
      }

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      this.currentSource = source;

      // Note: BufferSource.start is a scheduled estimate of playback start,
      // not physical audio output measured at speakers.
      if (!this.audibleReported && this.turnStartTime > 0) {
        this.audibleReported = true;
        const latency = Math.max(0, Date.now() - this.turnStartTime);
        if (this.onAudibleLatency) {
          this.onAudibleLatency(latency);
        }
      }

      source.onended = () => {
        if (this.sessionToken === token) {
          this.currentSource = null;
          this.onChunkEnded(chunk.turnId);
        }
      };

      source.start(0);
    } catch (err) {
      console.warn('Audio queue decode/playback error:', err);
      if (this.sessionToken === token) {
        this.currentSource = null;
        this.onChunkEnded(chunk.turnId);
      }
    }
  }

  simulateCurrentSourceEnd() {
    if (this.currentSource && typeof this.currentSource.onended === 'function') {
      const onEnded = this.currentSource.onended;
      this.currentSource = null;
      onEnded();
    }
  }

  onChunkEnded(turnId) {
    this.isPlaying = false;

    if (this.queue.length > 0) {
      this.processQueue();
    } else {
      if (this.turnCompleteReceived && this.pendingUtterancesCount === 0) {
        this._signalTurnEnd(turnId ?? this.currentTurnId);
      }
    }
  }

  interrupt() {
    this.sessionToken++;
    if (this.currentSource) {
      try {
        this.currentSource.stop();
        this.currentSource.disconnect();
      } catch {}
      this.currentSource = null;
    }
    this.queue = [];
    this.isPlaying = false;
    this.turnCompleteReceived = false;
    this.turnEndedReported = false;
    this.pendingUtterancesCount = 0;

    if (typeof window !== 'undefined' && window.speechSynthesis) {
      try {
        window.speechSynthesis.cancel();
      } catch {}
    }

    if (this.onSpeakingChange) {
      this.onSpeakingChange(false);
    }
  }

  speakClientText(text, turnId) {
    if (typeof window === 'undefined' || !window.speechSynthesis) {
      return false;
    }

    if (turnId !== undefined && turnId !== this.currentTurnId) {
      this.currentTurnId = turnId;
      this.turnCompleteReceived = false;
      this.turnEndedReported = false;
    }

    const token = this.sessionToken;
    this.isPlaying = true;
    this.pendingUtterancesCount = (this.pendingUtterancesCount || 0) + 1;

    if (this.onSpeakingChange) {
      this.onSpeakingChange(true);
    }

    const utterance = new window.SpeechSynthesisUtterance(text);
    utterance.lang = 'it-IT';

    // Latency measured on onstart (scheduled estimate of utterance speech start)
    utterance.onstart = () => {
      if (this.sessionToken === token && !this.audibleReported && this.turnStartTime > 0) {
        this.audibleReported = true;
        const latency = Math.max(0, Date.now() - this.turnStartTime);
        if (this.onAudibleLatency) {
          this.onAudibleLatency(latency);
        }
      }
    };

    const handleUtteranceDone = () => {
      if (this.sessionToken !== token) return;
      this.pendingUtterancesCount = Math.max(0, this.pendingUtterancesCount - 1);
      if (this.pendingUtterancesCount === 0) {
        this.isPlaying = false;
        if (this.turnCompleteReceived) {
          this._signalTurnEnd(turnId ?? this.currentTurnId);
        }
      }
    };

    utterance.onend = handleUtteranceDone;
    utterance.onerror = handleUtteranceDone;

    window.speechSynthesis.speak(utterance);
    return true;
  }

  isSpeaking() {
    return this.isPlaying || this.queue.length > 0 || this.pendingUtterancesCount > 0;
  }
}
