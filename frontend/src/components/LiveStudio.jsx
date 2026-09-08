import React, { useState, useEffect, useRef } from 'react';
import {
  Mic, MicOff, Volume2, Radio, Zap, Cpu, Activity, RotateCw,
  Play, Square, MessageSquare, AlertCircle, ShieldAlert,
  ExternalLink, ChevronDown, ChevronUp, Copy, Check, X,
  Sliders, Settings, Sparkles, MessageCircle
} from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';

const PROMPT_PRESETS = {
  concise: "Sei Alveare Live, assistente vocale ultra-rapido. Rispondi SEMPRE in modo brevissimo, diretto e conciso (massimo 1-2 frasi brevi, senza preamboli, senza elenchi). Parla in italiano naturale.",
  conversational: "Sei Alveare Live, assistente amichevole e colloquiale. Rispondi in italiano in modo chiaro e naturale in 2-3 frasi, adatto a una conversazione vocale fluida.",
  technical: "Sei Alveare, assistente AI esperto in tecnologia e programmazione. Rispondi in italiano in modo accurato, tecnico e sintetico.",
  custom: ""
};

// Convert Base64 string to ArrayBuffer for Web Audio decoding
export function base64ToArrayBuffer(base64) {
  const binaryString = typeof window !== 'undefined' ? window.atob(base64) : Buffer.from(base64, 'base64').toString('binary');
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

// Client-side downsampling from native mobile sample rate (e.g. 44.1kHz or 48kHz) to 16000Hz PCM16
export function downsampleTo16k(inputData, inputSampleRate) {
  if (!inputSampleRate || inputSampleRate === 16000) {
    const pcm16 = new Int16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
      const s = Math.max(-1, Math.min(1, inputData[i]));
      pcm16[i] = Math.round(s < 0 ? s * 0x8000 : s * 0x7FFF);
    }
    return pcm16;
  }

  const sampleRateRatio = inputSampleRate / 16000;
  const newLength = Math.round(inputData.length / sampleRateRatio);
  const result = new Int16Array(newLength);

  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < inputData.length; i++) {
      accum += inputData[i];
      count++;
    }
    const sample = count > 0 ? accum / count : inputData[Math.min(offsetBuffer, inputData.length - 1)];
    const s = Math.max(-1, Math.min(1, sample));
    result[offsetResult] = Math.round(s < 0 ? s * 0x8000 : s * 0x7FFF);
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

// Robust Web Audio decode helper supporting both Promises and callback signatures
export function decodeAudioBuffer(audioCtx, arrayBuffer) {
  return new Promise((resolve, reject) => {
    let resolved = false;
    const onOk = (buf) => {
      if (!resolved) { resolved = true; resolve(buf); }
    };
    const onErr = (err) => {
      if (!resolved) { resolved = true; reject(err); }
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

export default function LiveStudio({ apiBase, status, onNavigateToControl }) {
  const { t } = useTranslation();

  const [isLiveActive, setIsLiveActive] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected'); // 'disconnected' | 'connecting' | 'connected'
  const [assistantSpeaking, setAssistantSpeaking] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [interrupted, setInterrupted] = useState(false);

  // Insecure context (mobile LAN HTTP) & mic error state
  const checkIsInsecureContext = () => {
    if (typeof window === 'undefined') return false;
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]';
    const isHttp = protocol === 'http:';
    const hasGetUserMedia = Boolean(navigator?.mediaDevices?.getUserMedia);
    const isSecure = window.isSecureContext ?? !isHttp;
    return (!isSecure || !hasGetUserMedia) && isHttp && !isLocalhost;
  };

  const [isInsecureContext, setIsInsecureContext] = useState(checkIsInsecureContext);
  const [showChromeGuide, setShowChromeGuide] = useState(false);
  const [copiedField, setCopiedField] = useState(null);
  const [micError, setMicError] = useState(null);

  useEffect(() => {
    setIsInsecureContext(checkIsInsecureContext());
  }, []);

  const handleCopy = (text, field) => {
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(text);
      setCopiedField(field);
      setTimeout(() => setCopiedField(null), 2000);
    }
  };

  // Latency telemetry HUD
  const [sttLatency, setSttLatency] = useState(0);
  const [ttft, setTtft] = useState(0);
  const [ttfa, setTtfa] = useState(0);
  const [e2eLatency, setE2eLatency] = useState(0);

  // Conversation transcripts: array of { id, role: 'user' | 'assistant', text, metrics, timestamp }
  const [messages, setMessages] = useState([]);
  const [streamingAssistantText, setStreamingAssistantText] = useState('');

  // Voice preset selection
  const [selectedVoice, setSelectedVoice] = useState('if_sara');

  // Recommended Live preset toggle (Gemma 4 E4B on GPU)
  const [ultraFastLive, setUltraFastLive] = useState(true);

  // Mobile collapsible telemetry HUD
  const [showLatencyHud, setShowLatencyHud] = useState(true);

  // Tunable Live Settings & History Panel
  const [showSettingsDrawer, setShowSettingsDrawer] = useState(false);
  const [settingsNotice, setSettingsNotice] = useState(false);
  const [liveSettings, setLiveSettings] = useState(() => {
    try {
      const saved = localStorage.getItem('alveare_live_settings');
      if (saved) return JSON.parse(saved);
    } catch {}
    return {
      maxContextTurns: 4,
      maxTokens: 120,
      temperature: 0.5,
      speed: 1.0,
      voice: 'if_sara',
      minSilenceDuration: 0.35,
      enableThinking: false,
      systemPromptPreset: 'concise',
      customSystemPrompt: ''
    };
  });

  const slots = status?.slots || {};
  const llmSlot = slots.llm || {};
  const sttSlot = slots.stt || {};
  const ttsSlot = slots.tts || {};

  const isKokoroActive = Boolean(
    ttsSlot?.model?.toLowerCase().includes('kokoro') ||
    ttsSlot?.model?.toLowerCase().includes('cocore') ||
    ttsSlot?.model?.includes('82m')
  );

  const handleUpdateSetting = (key, value) => {
    setLiveSettings(prev => {
      const updated = { ...prev, [key]: value };
      try {
        localStorage.setItem('alveare_live_settings', JSON.stringify(updated));
      } catch {}

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const sysPrompt = updated.systemPromptPreset === 'custom'
          ? updated.customSystemPrompt
          : PROMPT_PRESETS[updated.systemPromptPreset] || PROMPT_PRESETS.concise;

        wsRef.current.send(JSON.stringify({
          type: 'config',
          settings: {
            max_context_turns: updated.maxContextTurns,
            max_tokens: updated.maxTokens,
            temperature: updated.temperature,
            voice: updated.voice,
            speed: updated.speed,
            min_silence_duration_s: updated.minSilenceDuration,
            enable_thinking: updated.enableThinking,
            system_prompt: sysPrompt
          }
        }));
        setSettingsNotice(true);
        setTimeout(() => setSettingsNotice(false), 2500);
      }
      return updated;
    });
  };

  // Audio & WebSocket refs
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const currentSourceRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingAudioRef = useRef(false);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const visualizerDataRef = useRef(new Uint8Array(32));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      if (!canvas || !ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const data = visualizerDataRef.current;
      const barCount = 24;
      const barWidth = (width / barCount) - 4;

      for (let i = 0; i < barCount; i++) {
        let val = data[i % data.length] || 0;
        if (isLiveActive) {
          if (userSpeaking) {
            val = Math.max(val, Math.sin(Date.now() * 0.01 + i) * 60 + 90);
          } else if (assistantSpeaking) {
            val = Math.max(val, Math.cos(Date.now() * 0.015 + i * 0.8) * 80 + 110);
          } else {
            val = Math.max(10, Math.sin(Date.now() * 0.003 + i * 0.5) * 15 + 20);
          }
        } else {
          val = 8;
        }

        const barHeight = Math.min(height - 6, (val / 255) * height);
        const x = i * (barWidth + 4) + 2;
        const y = height - barHeight;

        let fillStyle = '#a855f7';
        if (typeof ctx.createLinearGradient === 'function') {
          const gradient = ctx.createLinearGradient(0, height, 0, 0);
          if (assistantSpeaking) {
            gradient.addColorStop(0, '#10b981');
            gradient.addColorStop(1, '#06b6d4');
          } else if (userSpeaking) {
            gradient.addColorStop(0, '#f59e0b');
            gradient.addColorStop(1, '#ef4444');
          } else {
            gradient.addColorStop(0, 'rgba(168, 85, 247, 0.4)');
            gradient.addColorStop(1, 'rgba(168, 85, 247, 0.9)');
          }
          fillStyle = gradient;
        }

        ctx.fillStyle = fillStyle;
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(x, y, barWidth, barHeight, [4, 4, 0, 0]);
        } else {
          ctx.rect(x, y, barWidth, barHeight);
        }
        ctx.fill();
      }
    };

    draw();

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isLiveActive, userSpeaking, assistantSpeaking]);

  // Audio chunk playback queue handler using Web Audio API (mobile autoplay safe)
  const playNextAudioChunk = async () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingAudioRef.current = false;
      currentSourceRef.current = null;
      setAssistantSpeaking(false);
      return;
    }

    const audioCtx = audioContextRef.current;
    const audioB64 = audioQueueRef.current.shift();
    if (!audioB64) {
      playNextAudioChunk();
      return;
    }

    isPlayingAudioRef.current = true;
    setAssistantSpeaking(true);

    if (audioCtx && typeof audioCtx.decodeAudioData === 'function') {
      try {
        if (audioCtx.state === 'suspended') {
          await audioCtx.resume().catch(() => {});
        }

        const arrayBuffer = base64ToArrayBuffer(audioB64);
        const audioBuffer = await decodeAudioBuffer(audioCtx, arrayBuffer);

        // If stopped/interrupted while decoding, cancel playback
        if (!isPlayingAudioRef.current) {
          return;
        }

        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioCtx.destination);
        currentSourceRef.current = source;

        source.onended = () => {
          if (currentSourceRef.current === source) {
            currentSourceRef.current = null;
            playNextAudioChunk();
          }
        };

        source.start(0);
      } catch (err) {
        console.warn('Web Audio decode/playback failed, trying next chunk:', err);
        currentSourceRef.current = null;
        playNextAudioChunk();
      }
    } else {
      // Fallback for environments where Web Audio AudioContext is unavailable
      try {
        const audio = new Audio(`data:audio/wav;base64,${audioB64}`);
        audio.onended = () => playNextAudioChunk();
        audio.onerror = () => playNextAudioChunk();
        audio.play().catch(() => playNextAudioChunk());
      } catch {
        playNextAudioChunk();
      }
    }
  };

  const stopAllPlayback = () => {
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
        currentSourceRef.current.disconnect();
      } catch {}
      currentSourceRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    setAssistantSpeaking(false);
  };

  // Start Live Session
  const handleStartLive = async () => {
    try {
      setMicError(null);
      setConnectionStatus('connecting');
      setInterrupted(false);

      // Unlock Web Audio AudioContext synchronously inside user click gesture (bypasses mobile autoplay blocks)
      const AudioContextClass = typeof window !== 'undefined' ? (window.AudioContext || window.webkitAudioContext) : null;
      if (AudioContextClass) {
        if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
          audioContextRef.current = new AudioContextClass();
        }
        if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
          audioContextRef.current.resume().catch(() => {});
        }
      }

      // 1. Trigger live backend pipeline initialization if not already running
      try {
        await fetch(`${apiBase}/api/live/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            llm_model: llmSlot.model || 'gemma4-e4b',
            llm_device: llmSlot.device || 'gpu',
            stt_model: sttSlot.model || 'openai/whisper-large-v3-turbo',
            stt_device: sttSlot.device || 'cpu',
            tts_model: ttsSlot.model || 'hexgrad/Kokoro-82M',
            tts_device: ttsSlot.device || (isKokoroActive ? 'cpu' : 'cpu'),
            voice: liveSettings.voice || 'if_sara',
            max_context_turns: liveSettings.maxContextTurns,
            max_tokens: liveSettings.maxTokens,
            temperature: liveSettings.temperature,
            speed: liveSettings.speed,
            enable_thinking: liveSettings.enableThinking
          })
        });
      } catch (e) {
        console.warn('api/live/start notice:', e);
      }

      // 2. Open WebSocket connection
      const wsUrl = apiBase.replace(/^http/, 'ws') + '/ws/live';
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = async () => {
        setConnectionStatus('connected');
        setIsLiveActive(true);

        // 3. Request user microphone
        try {
          if (!navigator?.mediaDevices?.getUserMedia) {
            throw new Error(
              window.isSecureContext === false
                ? 'INSECURE_CONTEXT'
                : 'MEDIA_DEVICES_NOT_SUPPORTED'
            );
          }

          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: 16000,
              echoCancellation: true,
              noiseSuppression: true
            }
          });
          mediaStreamRef.current = stream;

          const AudioContextClass = window.AudioContext || window.webkitAudioContext;
          if (AudioContextClass) {
            let audioCtx = audioContextRef.current;
            if (!audioCtx || audioCtx.state === 'closed') {
              audioCtx = new AudioContextClass();
              audioContextRef.current = audioCtx;
            }
            if (audioCtx.state === 'suspended') {
              await audioCtx.resume().catch(() => {});
            }
            const source = audioCtx.createMediaStreamSource(stream);

            // ScriptProcessor node to capture raw PCM and resample to 16kHz
            const processor = audioCtx.createScriptProcessor(2048, 1, 1);
            source.connect(processor);
            processor.connect(audioCtx.destination);

            processor.onaudioprocess = (e) => {
              if (ws.readyState !== WebSocket.OPEN) return;
              // Mute output buffer to prevent mic feedback into speaker
              if (e.outputBuffer && typeof e.outputBuffer.getChannelData === 'function') {
                const outputData = e.outputBuffer.getChannelData(0);
                outputData.fill(0);
              }

              const inputData = e.inputBuffer.getChannelData(0);
              const actualSampleRate = audioCtx.sampleRate || e.inputBuffer.sampleRate || 16000;

              // Downsample from native phone rate (44.1k/48k) to 16000Hz before sending to WebSocket
              const pcm16 = downsampleTo16k(inputData, actualSampleRate);

              ws.send(pcm16.buffer);
            };
          }
        } catch (micErr) {
          console.warn('Microphone access not available or denied:', micErr);
          let errorMsg = t('liveStudio.micErrorGeneric', { error: micErr.message || String(micErr) });
          if (micErr.name === 'NotAllowedError' || micErr.name === 'PermissionDeniedError') {
            errorMsg = t('liveStudio.micErrorDenied');
          } else if (micErr.name === 'NotFoundError' || micErr.name === 'DevicesNotFoundError') {
            errorMsg = t('liveStudio.micErrorNotFound');
          } else if (micErr.message === 'INSECURE_CONTEXT' || window.isSecureContext === false || !navigator?.mediaDevices?.getUserMedia) {
            errorMsg = t('liveStudio.micErrorInsecure');
          }
          setMicError(errorMsg);
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const ev = data.event;

          if (ev === 'vad_speech_start') {
            setUserSpeaking(true);
            stopAllPlayback();
          } else if (ev === 'vad_speech_end') {
            setUserSpeaking(false);
          } else if (ev === 'interrupted') {
            setInterrupted(true);
            stopAllPlayback();
            setTimeout(() => setInterrupted(false), 2000);
          } else if (ev === 'user_transcript') {
            setUserSpeaking(false);
            setSttLatency(data.stt_latency_ms || 0);
            setMessages(prev => [
              ...prev,
              {
                id: `user-${Date.now()}`,
                role: 'user',
                text: data.text,
                stt_latency_ms: data.stt_latency_ms,
                timestamp: new Date().toLocaleTimeString()
              }
            ]);
            setStreamingAssistantText('');
          } else if (ev === 'ttft') {
            setTtft(data.ttft_ms || 0);
          } else if (ev === 'llm_chunk') {
            setStreamingAssistantText(prev => (prev ? prev + ' ' + data.text : data.text));
          } else if (ev === 'audio_chunk') {
            if (data.ttfa_ms) {
              setTtfa(data.ttfa_ms);
            }
            if (data.audio_b64) {
              audioQueueRef.current.push(data.audio_b64);
              if (!isPlayingAudioRef.current) {
                playNextAudioChunk();
              }
            }
          } else if (ev === 'settings_updated') {
            setSettingsNotice(true);
            setTimeout(() => setSettingsNotice(false), 2000);
          } else if (ev === 'turn_complete') {
            const metrics = data.metrics || {};
            setE2eLatency(metrics.e2e_latency_ms || 0);
            setMessages(prev => [
              ...prev,
              {
                id: `assistant-${Date.now()}`,
                role: 'assistant',
                text: data.full_text || streamingAssistantText,
                metrics: metrics,
                timestamp: new Date().toLocaleTimeString()
              }
            ]);
            setStreamingAssistantText('');
          }
        } catch (e) {
          console.error('Error parsing live WS event:', e);
        }
      };

      ws.onclose = () => {
        handleStopLive();
      };

      ws.onerror = (e) => {
        console.error('Live WS error:', e);
        handleStopLive();
      };

    } catch (e) {
      console.error('Live start error:', e);
      handleStopLive();
    }
  };

  const handleStopLive = () => {
    setIsLiveActive(false);
    setConnectionStatus('disconnected');
    setUserSpeaking(false);
    stopAllPlayback();

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch {}
      audioContextRef.current = null;
    }
  };

  return (
    <div className="responsive-page-container" style={{ flex: 1, display: 'flex', flexDirection: 'column', maxWidth: '1200px', margin: '0 auto', width: '100%', gap: '1.25rem', overflowX: 'hidden' }}>
      
      {/* INSECURE CONTEXT WARNING BANNER FOR MOBILE LAN ACCESS */}
      {isInsecureContext && (
        <div
          className="glass-card"
          role="alert"
          style={{
            padding: '1.25rem',
            background: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.4)',
            borderLeft: '5px solid #f59e0b',
            borderRadius: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.85rem'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.85rem' }}>
            <ShieldAlert size={26} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div style={{ flex: 1 }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 800, color: '#f59e0b', margin: 0, marginBottom: '0.35rem' }}>
                {t('liveStudio.insecureContextTitle')}
              </h3>
              <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.45' }}>
                {t('liveStudio.insecureContextDesc')}
              </p>
            </div>
          </div>

          {/* Action buttons & note */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', paddingLeft: '2.5rem' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
              <a
                href={`https://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8443`}
                className="btn-primary mobile-touch-target"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.6rem 1.25rem',
                  fontSize: '0.88rem',
                  fontWeight: 700,
                  textDecoration: 'none',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: 'white',
                  borderRadius: '8px',
                  minHeight: '44px'
                }}
              >
                <ExternalLink size={16} />
                {t('liveStudio.switchToHttps')}
              </a>

              <button
                type="button"
                onClick={() => setShowChromeGuide(prev => !prev)}
                className="btn-secondary mobile-touch-target"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.55rem 1rem',
                  fontSize: '0.82rem',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  minHeight: '44px'
                }}
              >
                {showChromeGuide ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                {t('liveStudio.chromeFlagsToggle')}
              </button>
            </div>

            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              {t('liveStudio.httpsSelfSignedNotice')}
            </div>

            {/* Expandable Chrome Android Guide */}
            {showChromeGuide && (
              <div
                style={{
                  marginTop: '0.5rem',
                  padding: '1rem',
                  background: 'rgba(0, 0, 0, 0.4)',
                  borderRadius: '8px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  fontSize: '0.82rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.6rem'
                }}
              >
                <div style={{ fontWeight: 700, color: '#f59e0b' }}>
                  {t('liveStudio.chromeFlagsTitle')}
                </div>

                <div>
                  <div>1. {t('liveStudio.chromeFlagsStep1')}</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
                    <code style={{ background: 'rgba(255,255,255,0.08)', padding: '0.25rem 0.5rem', borderRadius: '4px', color: '#67e8f9', wordBreak: 'break-all' }}>
                      chrome://flags/#unsafely-treat-insecure-origin-as-secure
                    </code>
                    <button
                      type="button"
                      className="btn-secondary mobile-touch-target"
                      onClick={() => handleCopy('chrome://flags/#unsafely-treat-insecure-origin-as-secure', 'flags')}
                      style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem', minHeight: '32px' }}
                    >
                      {copiedField === 'flags' ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                      <span>{copiedField === 'flags' ? t('liveStudio.copied') : t('liveStudio.copy')}</span>
                    </button>
                  </div>
                </div>

                <div>
                  <div>2. {t('liveStudio.chromeFlagsStep2')}</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
                    <code style={{ background: 'rgba(255,255,255,0.08)', padding: '0.25rem 0.5rem', borderRadius: '4px', color: '#67e8f9', wordBreak: 'break-all' }}>
                      {typeof window !== 'undefined' ? window.location.origin : 'http://192.168.132.197:8080'}
                    </code>
                    <button
                      type="button"
                      className="btn-secondary mobile-touch-target"
                      onClick={() => handleCopy(typeof window !== 'undefined' ? window.location.origin : 'http://192.168.132.197:8080', 'origin')}
                      style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem', minHeight: '32px' }}
                    >
                      {copiedField === 'origin' ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                      <span>{copiedField === 'origin' ? t('liveStudio.copied') : t('liveStudio.copy')}</span>
                    </button>
                  </div>
                </div>

                <div>
                  3. {t('liveStudio.chromeFlagsStep3')}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* MICROPHONE ERROR ALERT BANNER */}
      {micError && (
        <div
          className="glass-card"
          role="alert"
          style={{
            padding: '1rem 1.25rem',
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            borderLeft: '5px solid #ef4444',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <AlertCircle size={22} color="#ef4444" style={{ flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#ef4444' }}>
                {t('liveStudio.micErrorTitle')}
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                {micError}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setMicError(null)}
            className="btn-secondary mobile-touch-target"
            style={{ padding: '0.4rem', borderRadius: '50%', minHeight: '36px', minWidth: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            aria-label={t('liveStudio.dismiss')}
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* 1. TOP TRI-HARDWARE CONCURRENT ALLOCATION BANNER */}
      <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ flex: '1 1 300px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.3rem', flexWrap: 'wrap' }}>
            <Radio size={22} color="var(--accent-purple)" className={isLiveActive ? "pulse-icon" : ""} />
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800 }}>{t('liveStudio.title')}</h2>
            <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#c084fc', border: '1px solid rgba(168, 85, 247, 0.4)', fontWeight: 700 }}>
              {t('liveStudio.badge')}
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: '1.4' }}>
            {t('liveStudio.subtitle')}
          </p>
        </div>

        {/* Live Status Indicators */}
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {interrupted && (
            <span className="badge badge-danger" style={{ animation: 'pulse 1s infinite' }}>
              {t('liveStudio.bargeIn')}
            </span>
          )}

          <button
            type="button"
            className="btn-secondary mobile-touch-target"
            onClick={() => setShowSettingsDrawer(prev => !prev)}
            aria-label={t('liveStudio.liveSettings')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              padding: '0.75rem 1.1rem',
              minHeight: '48px',
              fontSize: '0.88rem',
              fontWeight: 600,
              borderColor: showSettingsDrawer ? 'var(--accent-cyan)' : 'var(--border-color)',
              background: showSettingsDrawer ? 'rgba(6, 182, 212, 0.15)' : 'rgba(0,0,0,0.25)',
              color: showSettingsDrawer ? 'var(--accent-cyan)' : 'var(--text-main)'
            }}
          >
            <Sliders size={17} />
            <span>{t('liveStudio.liveSettings')}</span>
            {settingsNotice && (
              <span className="badge badge-success" style={{ padding: '0.15rem 0.4rem', fontSize: '0.65rem' }}>
                ✓
              </span>
            )}
          </button>

          {isLiveActive ? (
            <button className="btn-danger mobile-touch-target" onClick={handleStopLive} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.4rem', minHeight: '48px', minWidth: '48px', fontSize: '0.9rem', fontWeight: 700 }}>
              <Square size={18} /> {t('liveStudio.stopSession')}
            </button>
          ) : (
            <button className="btn-primary mobile-touch-target" onClick={handleStartLive} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.4rem', minHeight: '48px', minWidth: '48px', fontSize: '0.9rem', fontWeight: 700, background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}>
              <Play size={18} /> {t('liveStudio.startSession')}
            </button>
          )}
        </div>
      </div>

      {/* LIVE SETTINGS DRAWER / PANEL */}
      {showSettingsDrawer && (
        <div
          className="glass-card"
          style={{
            padding: '1.25rem 1.5rem',
            borderRadius: '12px',
            border: '1px solid rgba(6, 182, 212, 0.4)',
            background: 'rgba(15, 23, 42, 0.85)',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Sliders size={20} color="var(--accent-cyan)" />
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
                {t('liveStudio.liveSettings')}
              </h3>
              {isLiveActive && (
                <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>
                  🟢 {t('liveStudio.liveSettingsUpdated')}
                </span>
              )}
            </div>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setShowSettingsDrawer(false)}
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
            >
              <X size={15} /> {t('liveStudio.closeSettings')}
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))', gap: '1.25rem' }}>
            
            {/* 1. Context Window History Turns */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  🧠 {t('liveStudio.contextWindow')}
                </label>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#38bdf8' }}>
                  {liveSettings.maxContextTurns} turni (~{liveSettings.maxContextTurns * 2} messaggi)
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={liveSettings.maxContextTurns}
                onChange={e => handleUpdateSetting('maxContextTurns', Number(e.target.value))}
                style={{ width: '100%', accentColor: '#38bdf8' }}
              />
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                {t('liveStudio.contextWindowDesc')}
              </span>
            </div>

            {/* 2. Max Tokens per Response */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  📝 {t('liveStudio.maxTokens')}
                </label>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#a78bfa' }}>
                  {liveSettings.maxTokens} token
                </span>
              </div>
              <input
                type="range"
                min="40"
                max="512"
                step="10"
                value={liveSettings.maxTokens}
                onChange={e => handleUpdateSetting('maxTokens', Number(e.target.value))}
                style={{ width: '100%', accentColor: '#a78bfa' }}
              />
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                Lunghezza massima di ciascuna risposta vocale generata dall'LLM.
              </span>
            </div>

            {/* 3. LLM Temperature */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  🌡️ {t('liveStudio.temperature')}
                </label>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#f59e0b' }}>
                  {liveSettings.temperature.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.5"
                step="0.05"
                value={liveSettings.temperature}
                onChange={e => handleUpdateSetting('temperature', parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#f59e0b' }}
              />
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                Bassa (0.2-0.5) per risposte precise e concise; alta (0.8+) per risposte più creative.
              </span>
            </div>

            {/* 4. Live Voice Selection */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                🎙️ {t('liveStudio.liveVoice')}
              </label>
              <select
                value={liveSettings.voice}
                onChange={e => handleUpdateSetting('voice', e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '8px',
                  background: 'rgba(0,0,0,0.4)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-main)',
                  fontSize: '0.85rem'
                }}
              >
                {isKokoroActive ? (
                  <>
                    <option value="if_sara">if_sara (Italiana Femminile Espressiva)</option>
                    <option value="im_nicola">im_nicola (Italiano Maschile Naturale)</option>
                    <option value="af_heart">af_heart (HeartMuLa Melodica)</option>
                  </>
                ) : (
                  <>
                    <option value="valeria">valeria (Ufficiale Alveare Live)</option>
                    <option value="daino">daino (Italiano Maschile)</option>
                    <option value="bolt">bolt (Assistente Conciso)</option>
                  </>
                )}
              </select>
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                {isKokoroActive ? 'Motore Kokoro-82M attivo (RTF ~0.10x su CPU Zen 5).' : 'Motore Audio8 attivo.'}
              </span>
            </div>

            {/* 5. Speech Speed */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  ⏩ {t('liveStudio.speechSpeed')}
                </label>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#34d399' }}>
                  {liveSettings.speed.toFixed(2)}x
                </span>
              </div>
              <input
                type="range"
                min="0.75"
                max="1.5"
                step="0.05"
                value={liveSettings.speed}
                onChange={e => handleUpdateSetting('speed', parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#34d399' }}
              />
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                Velocità di riproduzione della sintesi vocale.
              </span>
            </div>

            {/* 6. VAD Silence Duration */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  ⏱️ {t('liveStudio.vadSilence')}
                </label>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#60a5fa' }}>
                  {liveSettings.minSilenceDuration.toFixed(2)}s
                </span>
              </div>
              <input
                type="range"
                min="0.2"
                max="1.0"
                step="0.05"
                value={liveSettings.minSilenceDuration}
                onChange={e => handleUpdateSetting('minSilenceDuration', parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#60a5fa' }}
              />
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                {t('liveStudio.vadSilenceDesc')}
              </span>
            </div>

          </div>

          {/* 7. Thinking Toggle & System Prompt */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.9rem' }}>
            
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 600 }}>
                <input
                  type="checkbox"
                  checked={liveSettings.enableThinking}
                  onChange={e => handleUpdateSetting('enableThinking', e.target.checked)}
                  style={{ width: '18px', height: '18px', accentColor: 'var(--accent-purple)', cursor: 'pointer' }}
                />
                <span>🧠 {t('liveStudio.enableThinking')}</span>
              </label>
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                {t('liveStudio.enableThinkingDesc')}
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.4rem' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  💬 {t('liveStudio.systemPrompt')}
                </label>
                <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                  {[
                    { id: 'concise', label: t('liveStudio.presetConcise') },
                    { id: 'conversational', label: t('liveStudio.presetConversational') },
                    { id: 'technical', label: t('liveStudio.presetTechnical') },
                    { id: 'custom', label: t('liveStudio.presetCustom') }
                  ].map(p => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => handleUpdateSetting('systemPromptPreset', p.id)}
                      style={{
                        padding: '0.2rem 0.5rem',
                        fontSize: '0.72rem',
                        borderRadius: '6px',
                        border: liveSettings.systemPromptPreset === p.id ? '1px solid var(--accent-cyan)' : '1px solid var(--border-color)',
                        background: liveSettings.systemPromptPreset === p.id ? 'rgba(6, 182, 212, 0.2)' : 'rgba(0,0,0,0.3)',
                        color: liveSettings.systemPromptPreset === p.id ? 'var(--accent-cyan)' : 'var(--text-muted)',
                        cursor: 'pointer'
                      }}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              {liveSettings.systemPromptPreset === 'custom' ? (
                <textarea
                  rows="3"
                  value={liveSettings.customSystemPrompt}
                  onChange={e => handleUpdateSetting('customSystemPrompt', e.target.value)}
                  placeholder="Inserisci istruzioni di sistema personalizzate per l'assistente live..."
                  style={{
                    width: '100%',
                    padding: '0.6rem 0.75rem',
                    borderRadius: '8px',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-main)',
                    fontSize: '0.82rem',
                    fontFamily: 'var(--font-sans)',
                    resize: 'vertical'
                  }}
                />
              ) : (
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', background: 'rgba(0,0,0,0.25)', padding: '0.5rem 0.75rem', borderRadius: '6px', border: '1px solid var(--border-color)', fontStyle: 'italic' }}>
                  "{PROMPT_PRESETS[liveSettings.systemPromptPreset] || PROMPT_PRESETS.concise}"
                </div>
              )}
            </div>

          </div>
        </div>
      )}

      {/* ULTRA-FAST LIVE BANNER / BADGE */}
      <div
        className="glass-card"
        style={{
          padding: '0.85rem 1.25rem',
          background: 'linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 182, 212, 0.06) 100%)',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          borderRadius: '12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.75rem'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: '1 1 320px' }}>
          <span
            className="badge"
            style={{
              background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)',
              color: '#032014',
              fontWeight: 800,
              fontSize: '0.82rem',
              padding: '0.35rem 0.75rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}
          >
            <Zap size={14} /> {t('liveStudio.ultraFastBadge')}
          </span>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
            {t('liveStudio.ultraFastRecommendation')}
          </span>
        </div>

        <button
          type="button"
          onClick={() => setUltraFastLive(prev => !prev)}
          className="mobile-touch-target"
          style={{
            padding: '0.4rem 0.85rem',
            borderRadius: '20px',
            border: ultraFastLive ? '1px solid #10b981' : '1px solid var(--border-color)',
            background: ultraFastLive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(0,0,0,0.3)',
            color: ultraFastLive ? '#34d399' : 'var(--text-muted)',
            fontSize: '0.78rem',
            fontWeight: 700,
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.35rem'
          }}
          aria-label={ultraFastLive ? t('liveStudio.ultraFastToggleOn') : t('liveStudio.ultraFastToggleOff')}
        >
          <Zap size={13} />
          <span>{ultraFastLive ? t('liveStudio.ultraFastToggleOn') : t('liveStudio.ultraFastToggleOff')}</span>
        </button>
      </div>

      {/* 2. TRI-HARDWARE ROUTING CARDS */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
        {/* GPU: LLM Chat */}
        <div className="glass-card" style={{ padding: '1rem 1.25rem', borderLeft: '4px solid #c084fc', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#c084fc', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              🎮 {t('liveStudio.gpuCardTitle')}
            </span>
            <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>{t('liveStudio.gpuCardRole')}</span>
          </div>
          <div style={{ fontSize: '1rem', fontWeight: 800, wordBreak: 'break-all' }}>{llmSlot.model || 'gemma4-e4b'}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            {t('liveStudio.gpuCardDesc', { load: status?.gpu_usage?.percent || 0, tps: status?.tok_per_sec || 35 })}
          </div>
        </div>

        {/* NPU / CPU: Kokoro / Audio8 TTS */}
        <div className="glass-card" style={{ padding: '1rem 1.25rem', borderLeft: '4px solid #34d399', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#34d399', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              ⚡ {t('liveStudio.npuCardTitle')}
            </span>
            <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>
              {ttsSlot.device === 'npu' ? t('liveStudio.npuCardRole') : (sttSlot.device === 'npu' ? t('liveStudio.cpuCardRole') : t('liveStudio.npuCardRole'))}
            </span>
          </div>
          <div style={{ fontSize: '1rem', fontWeight: 800, wordBreak: 'break-all' }}>
            {ttsSlot.model || 'hexgrad/Kokoro-82M'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            {t('liveStudio.npuCardDesc')}
          </div>
        </div>

        {/* CPU: Whisper STT */}
        <div className="glass-card" style={{ padding: '1rem 1.25rem', borderLeft: '4px solid #60a5fa', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              🖥️ {t('liveStudio.cpuCardTitle')}
            </span>
            <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>
              {sttSlot.device === 'cpu' ? t('liveStudio.cpuCardRole') : (ttsSlot.device === 'cpu' ? t('liveStudio.npuCardRole') : t('liveStudio.cpuCardRole'))}
            </span>
          </div>
          <div style={{ fontSize: '1rem', fontWeight: 800, wordBreak: 'break-all' }}>
            {sttSlot.model || 'openai/whisper-large-v3-turbo'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            {t('liveStudio.cpuCardDesc')}
          </div>
        </div>
      </div>

      {/* 3. CENTRAL LIVE INTERACTIVE CONSOLE */}
      <div className="glass-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '1.25rem', gap: '1.25rem', minHeight: '380px' }}>
        
        {/* Visualizer and Mic Status Circle */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '1rem 0', gap: '1rem', width: '100%' }}>
          
          <div style={{ position: 'relative', width: '110px', height: '110px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div
              style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                background: assistantSpeaking
                  ? 'radial-gradient(circle, rgba(16, 185, 129, 0.4) 0%, rgba(16, 185, 129, 0) 70%)'
                  : userSpeaking
                  ? 'radial-gradient(circle, rgba(245, 158, 11, 0.4) 0%, rgba(245, 158, 11, 0) 70%)'
                  : isLiveActive
                  ? 'radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, rgba(168, 85, 247, 0) 70%)'
                  : 'transparent',
                animation: isLiveActive ? 'pulse 2s infinite' : 'none'
              }}
            />
            <button
              onClick={isLiveActive ? handleStopLive : handleStartLive}
              className="mobile-touch-target"
              aria-label={isLiveActive ? t('liveStudio.stopSession') : t('liveStudio.startSession')}
              style={{
                width: '80px',
                height: '80px',
                minWidth: '80px',
                minHeight: '80px',
                borderRadius: '50%',
                border: 'none',
                background: isLiveActive ? 'var(--gradient-brand)' : 'rgba(255,255,255,0.08)',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                boxShadow: isLiveActive ? 'var(--shadow-glow)' : 'none',
                transition: 'all 0.2s ease',
                zIndex: 2
              }}
            >
              {isLiveActive ? <Mic size={36} /> : <MicOff size={36} color="var(--text-muted)" />}
            </button>
          </div>

          <div style={{ textAlign: 'center', maxWidth: '500px', width: '100%' }}>
            <div style={{ fontWeight: 700, fontSize: '1.05rem' }}>
              {assistantSpeaking ? t('liveStudio.statusSpeaking') : userSpeaking ? t('liveStudio.statusListening') : isLiveActive ? t('liveStudio.statusReady') : t('liveStudio.statusInactive')}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: '1.4' }}>
              {isLiveActive ? t('liveStudio.descActive') : t('liveStudio.descInactive')}
            </div>
          </div>

          {/* Real-time Canvas Waveform */}
          <canvas
            ref={canvasRef}
            width={320}
            height={60}
            style={{ width: '100%', maxWidth: '340px', height: '60px', borderRadius: '8px', background: 'rgba(0,0,0,0.2)', display: 'block' }}
          />

          {/* Telemetry HUD Collapsible Toggle Button */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '0.25rem' }}>
            <button
              type="button"
              onClick={() => setShowLatencyHud(prev => !prev)}
              className="btn-secondary mobile-touch-target"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.45rem 1rem',
                fontSize: '0.8rem',
                borderRadius: '20px',
                cursor: 'pointer',
                minHeight: '44px'
              }}
              aria-expanded={showLatencyHud}
            >
              <Activity size={15} color="var(--accent-cyan)" />
              <span>{showLatencyHud ? t('liveStudio.hideMetrics') : t('liveStudio.showMetrics')}</span>
            </button>
          </div>

          {/* Real-Time Latency Telemetry HUD */}
          {showLatencyHud && (
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center', background: 'rgba(0,0,0,0.3)', padding: '0.75rem 1.25rem', borderRadius: '12px', border: '1px solid var(--border-color)', width: '100%', maxWidth: '640px' }}>
              <div style={{ textAlign: 'center', minWidth: '90px', flex: '1 1 90px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.sttLatency')}</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#34d399' }}>{sttLatency > 0 ? `${sttLatency} ms` : '—'}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: '120px', flex: '1 1 120px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.ttft')}</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#c084fc' }}>{ttft > 0 ? `${ttft} ms` : '—'}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: '120px', flex: '1 1 120px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.ttfa')}</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#60a5fa' }}>{ttfa > 0 ? `${ttfa} ms` : '—'}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: '100px', flex: '1 1 100px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.e2e')}</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#f59e0b' }}>{e2eLatency > 0 ? `${e2eLatency} ms` : '—'}</div>
              </div>
            </div>
          )}
        </div>

        {/* 4. STREAMING CONVERSATION LOG */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '350px', paddingRight: '0.35rem', width: '100%' }}>
          {messages.length === 0 && !streamingAssistantText ? (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem 0', fontSize: '0.9rem' }}>
              {t('liveStudio.emptyTranscript')}
            </div>
          ) : (
            messages.map(msg => (
              <div
                key={msg.id}
                style={{
                  alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '85%',
                  padding: '0.75rem 1rem',
                  borderRadius: '12px',
                  background: msg.role === 'user' ? 'var(--gradient-brand)' : 'rgba(255,255,255,0.06)',
                  color: 'white',
                  border: msg.role === 'user' ? 'none' : '1px solid var(--border-color)',
                  fontSize: '0.92rem',
                  wordBreak: 'break-word',
                  overflowWrap: 'break-word'
                }}
              >
                <div style={{ fontSize: '0.7rem', opacity: 0.7, marginBottom: '0.25rem', display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                  <span>{msg.role === 'user' ? 'You' : 'Alveare Live'}</span>
                  <span>{msg.timestamp}</span>
                </div>
                <div>{msg.text}</div>
                {msg.metrics?.e2e_latency_ms && (
                  <div style={{ fontSize: '0.68rem', opacity: 0.6, marginTop: '0.35rem' }}>
                    ⚡ Turnaround: {msg.metrics.e2e_latency_ms} ms (STT: {msg.metrics.stt_latency_ms} ms | TTFT: {msg.metrics.ttft_ms} ms)
                  </div>
                )}
              </div>
            ))
          )}

          {/* Currently streaming tokens */}
          {streamingAssistantText && (
            <div
              style={{
                alignSelf: 'flex-start',
                maxWidth: '85%',
                padding: '0.75rem 1rem',
                borderRadius: '12px',
                background: 'rgba(255,255,255,0.06)',
                color: 'white',
                border: '1px solid var(--border-color)',
                fontSize: '0.92rem',
                wordBreak: 'break-word',
                overflowWrap: 'break-word'
              }}
            >
              <div style={{ fontSize: '0.7rem', opacity: 0.7, marginBottom: '0.25rem' }}>Alveare Live (Reasoning...)</div>
              <div>{streamingAssistantText}</div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
