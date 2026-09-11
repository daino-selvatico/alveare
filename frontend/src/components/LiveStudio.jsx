import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Mic, MicOff, Volume2, VolumeX, Radio, Zap, Activity,
  Play, Square, AlertCircle, ShieldAlert,
  ExternalLink, ChevronDown, ChevronUp, Copy, Check, X,
  Sliders, Bot, Wrench, Trash2, StopCircle
} from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';
import { AudioPlaybackQueue, base64ToArrayBuffer, decodeAudioBuffer } from '../utils/audioQueue';

export { base64ToArrayBuffer, decodeAudioBuffer };

export const PROMPT_PRESETS = {
  concise: "Sei Alveare Live, assistente vocale ultra-rapido. Rispondi SEMPRE in modo brevissimo, diretto e conciso (massimo 1-2 frasi brevi, senza preamboli, senza elenchi). Parla in italiano naturale.",
  conversational: "Sei Alveare Live, assistente amichevole e colloquiale. Rispondi in italiano in modo chiaro e naturale in 2-3 frasi, adatto a una conversazione vocale fluida.",
  technical: "Sei Alveare, assistente AI esperto in tecnologia e programmazione. Rispondi in italiano in modo accurato, tecnico e sintetico.",
  custom: ""
};

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

export default function LiveStudio({ apiBase, status, onNavigateToControl = null }) {
  const { t } = useTranslation();

  // Mode: 'assistant' (Tap-to-talk, single turn) vs 'live' (Continuous full-duplex)
  const [mode, setMode] = useState('assistant');

  // Assistant states: 'idle' | 'listening' | 'processing' | 'speaking'
  const [assistantState, setAssistantState] = useState('idle');
  const assistantStateRef = useRef('idle');
  assistantStateRef.current = assistantState;

  // Live session state
  const [isLiveActive, setIsLiveActive] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected'); // 'disconnected' | 'connecting' | 'connected'
  const [isMuted, setIsMuted] = useState(false);
  const isMutedRef = useRef(false);
  isMutedRef.current = isMuted;

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
  const [errorTitle, setErrorTitle] = useState(null);

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

  // Latency telemetry HUD (Server TTFA vs Audible Latency)
  const [sttLatency, setSttLatency] = useState(0);
  const [ttft, setTtft] = useState(0);
  const [ttfa, setTtfa] = useState(0);
  const [audibleLatency, setAudibleLatency] = useState(0);
  const [e2eLatency, setE2eLatency] = useState(0);

  // Conversation transcripts: array of { id, role: 'user' | 'assistant' | 'tool', text, metrics, timestamp, name, args, status, result }
  const [messages, setMessages] = useState([]);
  const [streamingAssistantText, setStreamingAssistantText] = useState('');

  // Recommended Live preset toggle (Gemma 4 E4B on GPU)
  const [ultraFastLive, setUltraFastLive] = useState(true);

  // Collapsible telemetry HUD & Hardware details
  const [showLatencyHud, setShowLatencyHud] = useState(true);
  const [showHardwareDetails, setShowHardwareDetails] = useState(false);
  const [isMicPhysicallyActive, setIsMicPhysicallyActive] = useState(false);

  // Tunable Live Settings & History Panel
  const [showSettingsDrawer, setShowSettingsDrawer] = useState(false);
  const [settingsNotice, setSettingsNotice] = useState(false);
  const [clearMemoryNotice, setClearMemoryNotice] = useState(false);

  // Detection of client TTS capability
  const isClientTtsSupported = typeof window !== 'undefined' && 'speechSynthesis' in window && typeof window.SpeechSynthesisUtterance !== 'undefined';

  const [liveSettings, setLiveSettings] = useState(() => {
    try {
      const saved = localStorage.getItem('alveare_live_settings');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.ttsMode === 'client' && !isClientTtsSupported) {
          parsed.ttsMode = 'server';
        }
        return parsed;
      }
    } catch {}
    return {
      maxContextTurns: 4,
      maxTokens: 120,
      temperature: 0.5,
      speed: 1.0,
      voice: 'if_sara',
      ttsMode: 'server', // 'server' | 'client'
      minSilenceDuration: 0.35,
      enableThinking: false,
      systemPromptPreset: 'concise',
      customSystemPrompt: ''
    };
  });

  const liveSettingsRef = useRef(liveSettings);
  useEffect(() => {
    liveSettingsRef.current = liveSettings;
  }, [liveSettings]);

  // Sentence chunk buffer for client TTS mode
  const clientSentenceBufferRef = useRef('');
  const clientSentencesSpokenCountRef = useRef(0);

  // Bounded timers
  const clearMemoryTimerRef = useRef(null);
  const settingsNoticeTimerRef = useRef(null);
  const interruptedTimerRef = useRef(null);
  const copiedFieldTimerRef = useRef(null);

  const slots = status?.slots || {};
  const llmSlot = slots.llm || {};
  const sttSlot = slots.stt || {};
  const ttsSlot = slots.tts || {};

  const isKokoroActive = Boolean(
    ttsSlot?.model?.toLowerCase().includes('kokoro') ||
    ttsSlot?.model?.toLowerCase().includes('cocore') ||
    ttsSlot?.model?.includes('82m')
  );

  // Session refs & tokens to prevent stale async executions
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorNodeRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const sessionTokenRef = useRef(0);
  const turnIdCounterRef = useRef(0);
  const currentTurnIdRef = useRef(0);
  const turnStartTimeRef = useRef(0);

  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const visualizerDataRef = useRef(new Uint8Array(32));

  // Audio Playback Queue
  const audioQueueRef = useRef(null);

  const handleAudioPlaybackEnd = useCallback(() => {
    setAssistantSpeaking(false);
    if (assistantStateRef.current === 'speaking') {
      setAssistantState('idle');
    }
  }, []);

  const handleAudibleLatency = useCallback((lat) => {
    setAudibleLatency(lat);
  }, []);

  if (!audioQueueRef.current) {
    audioQueueRef.current = new AudioPlaybackQueue({
      getAudioContext: () => audioContextRef.current,
      ttsMode: liveSettings.ttsMode,
      onSpeakingChange: (speaking) => {
        setAssistantSpeaking(speaking);
        if (!speaking && assistantStateRef.current === 'speaking') {
          setAssistantState('idle');
        }
      },
      onTurnEnd: handleAudioPlaybackEnd,
      onAudibleLatency: handleAudibleLatency
    });
  }

  useEffect(() => {
    if (audioQueueRef.current) {
      audioQueueRef.current.setTtsMode(liveSettings.ttsMode);
    }
  }, [liveSettings.ttsMode]);

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
            tts_mode: updated.ttsMode,
            speed: updated.speed,
            min_silence_duration_s: updated.minSilenceDuration,
            enable_thinking: updated.enableThinking,
            system_prompt: sysPrompt
          }
        }));
        setSettingsNotice(true);
        clearTimeout(settingsNoticeTimerRef.current);
        settingsNoticeTimerRef.current = setTimeout(() => setSettingsNotice(false), 2500);
      }
      return updated;
    });
  };

  // Canvas visualizer animation
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
      const isActiveSession = mode === 'assistant' ? assistantState !== 'idle' : isLiveActive;

      for (let i = 0; i < barCount; i++) {
        let val = data[i % data.length] || 0;
        if (isActiveSession) {
          if (userSpeaking || assistantState === 'listening') {
            val = Math.max(val, Math.sin(Date.now() * 0.01 + i) * 60 + 90);
          } else if (assistantSpeaking || assistantState === 'speaking') {
            val = Math.max(val, Math.cos(Date.now() * 0.015 + i * 0.8) * 80 + 110);
          } else if (assistantState === 'processing') {
            val = Math.max(val, Math.sin(Date.now() * 0.008 + i) * 30 + 40);
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
          if (assistantSpeaking || assistantState === 'speaking') {
            gradient.addColorStop(0, '#10b981');
            gradient.addColorStop(1, '#06b6d4');
          } else if (userSpeaking || assistantState === 'listening') {
            gradient.addColorStop(0, '#f59e0b');
            gradient.addColorStop(1, '#ef4444');
          } else if (assistantState === 'processing') {
            gradient.addColorStop(0, '#8b5cf6');
            gradient.addColorStop(1, '#3b82f6');
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
  }, [mode, assistantState, isLiveActive, userSpeaking, assistantSpeaking]);

  // Clean teardown for mic capture only (preserves WS and AudioContext if needed)
  const stopMicCapture = useCallback(() => {
    if (processorNodeRef.current) {
      try {
        processorNodeRef.current.onaudioprocess = null;
        processorNodeRef.current.disconnect();
      } catch {}
      processorNodeRef.current = null;
    }

    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch {}
      sourceNodeRef.current = null;
    }

    if (mediaStreamRef.current) {
      try {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      } catch {}
      mediaStreamRef.current = null;
    }
    setIsMicPhysicallyActive(false);
  }, []);

  // Clean teardown for audio nodes, mic tracks, audio context, and WS
  const cleanupAllAudio = useCallback(() => {
    sessionTokenRef.current++;

    if (audioQueueRef.current) {
      audioQueueRef.current.interrupt();
    }

    stopMicCapture();

    if (audioContextRef.current) {
      try {
        if (audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
      } catch {}
      audioContextRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }

    setAssistantSpeaking(false);
    setUserSpeaking(false);
    setConnectionStatus('disconnected');
  }, [stopMicCapture]);

  // Teardown on unmount
  useEffect(() => {
    return () => {
      cleanupAllAudio();
      clearTimeout(clearMemoryTimerRef.current);
      clearTimeout(settingsNoticeTimerRef.current);
      clearTimeout(interruptedTimerRef.current);
      clearTimeout(copiedFieldTimerRef.current);
    };
  }, [cleanupAllAudio]);

  // Local audio invalidation only (called on server interrupted or VAD start: NEVER sends WS message)
  const handleLocalInterrupt = useCallback(() => {
    if (audioQueueRef.current) {
      audioQueueRef.current.interrupt();
    }
    clientSentenceBufferRef.current = '';
    clientSentencesSpokenCountRef.current = 0;
    setInterrupted(true);
    setAssistantSpeaking(false);
    if (assistantStateRef.current === 'speaking' || assistantStateRef.current === 'processing') {
      setAssistantState('idle');
    }
    clearTimeout(interruptedTimerRef.current);
    interruptedTimerRef.current = setTimeout(() => setInterrupted(false), 1500);
  }, []);

  // User-originated interruption / barge-in (sends interrupt event to backend)
  const handleUserInterrupt = useCallback(() => {
    handleLocalInterrupt();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
      } catch {}
    }
  }, [handleLocalInterrupt]);

  // Toggle Mute
  const handleToggleMute = () => {
    setIsMuted(prev => {
      const next = !prev;
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getAudioTracks().forEach(t => {
          t.enabled = !next;
        });
      }
      return next;
    });
  };

  // Clear memory / reset context
  const handleClearMemory = () => {
    handleLocalInterrupt();
    setMessages([]);
    setStreamingAssistantText('');

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'clear_memory' }));
      } catch {}
    } else {
      // Local immediate notice when socket is offline
      setClearMemoryNotice(true);
      clearTimeout(clearMemoryTimerRef.current);
      clearMemoryTimerRef.current = setTimeout(() => setClearMemoryNotice(false), 2500);
    }
  };

  // Setup WebSocket message dispatcher
  const setupWebSocket = (onReady) => {
    const wsUrl = apiBase.replace(/^http/, 'ws') + '/ws/live';
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      // Wait for server live_connected handshake before setting connected or sending initial config
      setConnectionStatus('connecting');
    };

    ws.onmessage = (event) => {
      if (wsRef.current !== ws) return;
      try {
        const data = JSON.parse(event.data);
        const ev = data.event;

        if (ev === 'live_connected') {
          setConnectionStatus('connected');
          const currentSettings = liveSettingsRef.current;
          const sysPrompt = currentSettings.systemPromptPreset === 'custom'
            ? currentSettings.customSystemPrompt
            : PROMPT_PRESETS[currentSettings.systemPromptPreset] || PROMPT_PRESETS.concise;

          // Apply initial configuration strictly AFTER live_connected handshake
          try {
            ws.send(JSON.stringify({
              type: 'config',
              settings: {
                max_context_turns: currentSettings.maxContextTurns,
                max_tokens: currentSettings.maxTokens,
                temperature: currentSettings.temperature,
                voice: currentSettings.voice,
                tts_mode: currentSettings.ttsMode,
                speed: currentSettings.speed,
                min_silence_duration_s: currentSettings.minSilenceDuration,
                enable_thinking: currentSettings.enableThinking,
                system_prompt: sysPrompt
              }
            }));
          } catch {}

          if (onReady) onReady();
        } else if (ev === 'vad_speech_start') {
          setUserSpeaking(true);
          // Only invalidate local audio; NEVER emit another interrupt back to server
          handleLocalInterrupt();
        } else if (ev === 'vad_speech_end') {
          setUserSpeaking(false);
          turnStartTimeRef.current = Date.now();
          if (audioQueueRef.current) {
            audioQueueRef.current.setTurnStartTime(turnStartTimeRef.current);
          }
          if (assistantStateRef.current === 'listening') {
            setAssistantState('processing');
            // Privacy protection: stop physical capture as soon as utterance ends
            stopMicCapture();
          }
        } else if (ev === 'interrupted') {
          // Server interrupt ACK: invalidate local playback only, NEVER emit another interrupt!
          handleLocalInterrupt();
        } else if (ev === 'user_transcript') {
          setUserSpeaking(false);
          setSttLatency(data.stt_latency_ms || 0);
          setMessages(prev => [
            ...prev,
            {
              id: `user-${Date.now()}-${Math.random()}`,
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
          setStreamingAssistantText(prev => (prev ? prev + data.text : data.text));
          if (liveSettingsRef.current.ttsMode === 'client' && isClientTtsSupported) {
            clientSentenceBufferRef.current += data.text;
            const sentenceMatch = clientSentenceBufferRef.current.match(/^([\s\S]*?[.!?\n]+)\s*([\s\S]*)$/);
            if (sentenceMatch) {
              const sentenceToSpeak = sentenceMatch[1].trim();
              clientSentenceBufferRef.current = sentenceMatch[2];
              if (sentenceToSpeak) {
                clientSentencesSpokenCountRef.current++;
                setAssistantSpeaking(true);
                if (assistantStateRef.current === 'processing') {
                  setAssistantState('speaking');
                }
                audioQueueRef.current?.speakClientText(sentenceToSpeak, currentTurnIdRef.current);
              }
            }
          }
        } else if (ev === 'audio_chunk') {
          if (data.ttfa_ms) {
            setTtfa(data.ttfa_ms);
          }
          if (data.audio_b64 && liveSettingsRef.current.ttsMode !== 'client') {
            setAssistantSpeaking(true);
            if (assistantStateRef.current === 'processing') {
              setAssistantState('speaking');
            }
            audioQueueRef.current?.enqueue({
              audioB64: data.audio_b64,
              turnId: data.turn_id ?? turnIdCounterRef.current,
              sampleRate: data.sample_rate,
              format: data.format
            });
          }
        } else if (ev === 'tool_call_start' || ev === 'tool_call') {
          const toolName = data.tool_name || data.tool || 'tool';
          const toolArgs = data.tool_args || data.args || (data.query ? { query: data.query } : (data.command ? { command: data.command } : undefined));
          const turnId = data.turn_id;

          setMessages(prev => {
            const alreadyExists = prev.some(m => m.role === 'tool' && m.name === toolName && m.status === 'running' && (turnId === undefined || m.turn_id === turnId));
            if (alreadyExists) return prev;
            return [
              ...prev,
              {
                id: `tool-${Date.now()}-${Math.random()}`,
                role: 'tool',
                name: toolName,
                args: toolArgs,
                status: 'running',
                turn_id: turnId,
                timestamp: new Date().toLocaleTimeString()
              }
            ];
          });
        } else if (ev === 'tool_call_done' || ev === 'tool_call_end') {
          const toolName = data.tool_name || data.tool || 'tool';
          const summary = data.result_summary || data.summary || data.result || '';
          const isError = Boolean(data.error) || data.status === 'error';
          const turnId = data.turn_id;

          setMessages(prev =>
            prev.map(msg => {
              if (msg.role === 'tool' && msg.name === toolName && msg.status === 'running' && (turnId === undefined || msg.turn_id === undefined || msg.turn_id === turnId)) {
                return {
                  ...msg,
                  status: isError ? 'error' : 'done',
                  result: summary,
                  summary: summary,
                  error: data.error
                };
              }
              return msg;
            })
          );
        } else if (ev === 'settings_updated') {
          setSettingsNotice(true);
          clearTimeout(settingsNoticeTimerRef.current);
          settingsNoticeTimerRef.current = setTimeout(() => setSettingsNotice(false), 2000);
        } else if (ev === 'memory_cleared') {
          setClearMemoryNotice(true);
          clearTimeout(clearMemoryTimerRef.current);
          clearMemoryTimerRef.current = setTimeout(() => setClearMemoryNotice(false), 2500);
        } else if (ev === 'error' || ev === 'tts_error') {
          console.error('Live audio error event:', data.error || data.message);
          const errText = String(data.error || data.message || '');
          const isTtsErr = ev === 'tts_error' || errText.includes('en_core_web_sm') || errText.toLowerCase().includes('tts') || errText.toLowerCase().includes('kokoro');
          setErrorTitle(isTtsErr ? 'Errore Sintesi Vocale (TTS)' : 'Errore Server Live');
          setMicError(data.error || data.message || 'Errore del server audio');
          if (assistantStateRef.current === 'processing') {
            setAssistantState('idle');
          }
        } else if (ev === 'turn_complete') {
          const metrics = data.metrics || {};
          setE2eLatency(metrics.e2e_latency_ms || 0);
          const turnId = data.turn_id ?? turnIdCounterRef.current++;
          currentTurnIdRef.current = turnId + 1;

          const finalizedText = data.full_text || streamingAssistantText;
          setStreamingAssistantText('');

          setMessages(prev => [
            ...prev,
            {
              id: `assistant-${Date.now()}-${Math.random()}`,
              role: 'assistant',
              text: finalizedText,
              metrics: metrics,
              timestamp: new Date().toLocaleTimeString()
            }
          ]);

          if (liveSettingsRef.current.ttsMode === 'client' && isClientTtsSupported) {
            const leftover = clientSentenceBufferRef.current.trim();
            clientSentenceBufferRef.current = '';
            if (leftover) {
              clientSentencesSpokenCountRef.current++;
              audioQueueRef.current?.speakClientText(leftover, turnId);
            } else if (clientSentencesSpokenCountRef.current === 0 && finalizedText) {
              clientSentencesSpokenCountRef.current++;
              audioQueueRef.current?.speakClientText(finalizedText, turnId);
            }
            clientSentencesSpokenCountRef.current = 0;
            audioQueueRef.current?.markTurnComplete(turnId);
          } else {
            audioQueueRef.current?.markTurnComplete(turnId);
          }
        }
      } catch (e) {
        console.error('Error parsing live WS event:', e);
      }
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      stopMicCapture();
      setConnectionStatus('disconnected');
      setIsLiveActive(false);
      if (assistantStateRef.current !== 'idle') {
        setAssistantState('idle');
      }
    };

    ws.onerror = (e) => {
      if (wsRef.current !== ws) return;
      console.error('Live WS error:', e);
      stopMicCapture();
      setConnectionStatus('disconnected');
      setIsLiveActive(false);
      setErrorTitle('Errore Connessione Live');
      setMicError(t('liveStudio.wsError') || 'Errore di connessione WebSocket');
      if (assistantStateRef.current !== 'idle') {
        setAssistantState('idle');
      }
    };

    return ws;
  };

  // Helper to acquire user microphone
  const acquireMicrophone = async () => {
    if (!navigator?.mediaDevices?.getUserMedia) {
      throw new Error(
        window.isSecureContext === false
          ? 'INSECURE_CONTEXT'
          : 'MEDIA_DEVICES_NOT_SUPPORTED'
      );
    }

    const currentToken = sessionTokenRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    if (sessionTokenRef.current !== currentToken) {
      stream.getTracks().forEach(t => t.stop());
      return null;
    }

    mediaStreamRef.current = stream;
    setIsMicPhysicallyActive(true);

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
      sourceNodeRef.current = source;

      const processor = audioCtx.createScriptProcessor(2048, 1, 1);
      processorNodeRef.current = processor;
      source.connect(processor);
      processor.connect(audioCtx.destination);

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        if (isMutedRef.current) return;

        // Mute output buffer to prevent mic feedback into speaker
        if (e.outputBuffer && typeof e.outputBuffer.getChannelData === 'function') {
          const outputData = e.outputBuffer.getChannelData(0);
          outputData.fill(0);
        }

        const inputData = e.inputBuffer.getChannelData(0);
        const actualSampleRate = audioCtx.sampleRate || e.inputBuffer.sampleRate || 16000;
        const pcm16 = downsampleTo16k(inputData, actualSampleRate);

        wsRef.current.send(pcm16.buffer);
      };
    }
    return stream;
  };

  // Assistente: Tap-to-Talk primary handler
  const handleAssistantTap = async () => {
    if (assistantState === 'idle') {
      try {
        setMicError(null);
        setAssistantState('listening');

        // Unlock audio context on user gesture
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass && (!audioContextRef.current || audioContextRef.current.state === 'closed')) {
          audioContextRef.current = new AudioContextClass();
        }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          setupWebSocket();
        }
        await acquireMicrophone();
      } catch (err) {
        handleMicError(err);
        setAssistantState('idle');
      }
    } else if (assistantState === 'listening') {
      // User tapped to finish utterance
      turnStartTimeRef.current = Date.now();
      if (audioQueueRef.current) {
        audioQueueRef.current.setTurnStartTime(turnStartTimeRef.current);
      }
      setAssistantState('processing');

      // Stop mic capture immediately to preserve privacy
      stopMicCapture();
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ type: 'end_of_utterance' }));
        } catch {}
      }
    } else if (assistantState === 'processing' || assistantState === 'speaking') {
      // User tapped to interrupt
      handleUserInterrupt();
    }
  };

  // Live: Start session
  const handleStartLive = async () => {
    try {
      setMicError(null);
      setConnectionStatus('connecting');
      setInterrupted(false);

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass && (!audioContextRef.current || audioContextRef.current.state === 'closed')) {
        audioContextRef.current = new AudioContextClass();
      }

      // 1. Start live backend pipeline
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
            tts_mode: liveSettings.ttsMode,
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

      // 2. Open WS & acquire mic
      setupWebSocket();
      setIsLiveActive(true);
      try {
        await acquireMicrophone();
      } catch (micErr) {
        handleMicError(micErr);
      }
    } catch (e) {
      console.error('Live start error:', e);
      handleStopLive();
    }
  };

  // Live: Stop session
  const handleStopLive = () => {
    setIsLiveActive(false);
    cleanupAllAudio();
  };

  const handleMicError = (micErr) => {
    console.warn('Microphone access error:', micErr);
    setErrorTitle(t('liveStudio.micErrorTitle'));
    let errorMsg = t('liveStudio.micErrorGeneric', { error: micErr.message || String(micErr) });
    if (micErr.name === 'NotAllowedError' || micErr.name === 'PermissionDeniedError') {
      errorMsg = t('liveStudio.micErrorDenied');
    } else if (micErr.name === 'NotFoundError' || micErr.name === 'DevicesNotFoundError') {
      errorMsg = t('liveStudio.micErrorNotFound');
    } else if (micErr.message === 'INSECURE_CONTEXT' || window.isSecureContext === false || !navigator?.mediaDevices?.getUserMedia) {
      errorMsg = t('liveStudio.micErrorInsecure');
    }
    setMicError(errorMsg);
  };

  // Switch between Assistente and Live mode
  const handleSwitchMode = (newMode) => {
    if (newMode === mode) return;
    cleanupAllAudio();
    setMode(newMode);
    setAssistantState('idle');
    setIsLiveActive(false);
    setStreamingAssistantText('');
  };

  // Keyboard navigation: Space/Enter on primary button, Esc to interrupt
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      handleUserInterrupt();
    }
  };

  return (
    <div
      className="responsive-page-container"
      onKeyDown={handleKeyDown}
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        maxWidth: '1200px',
        margin: '0 auto',
        width: '100%',
        gap: 'clamp(0.65rem, 1.8vw, 1.25rem)',
        overflowX: 'hidden',
        boxSizing: 'border-box'
      }}
    >
      {/* INSECURE CONTEXT WARNING BANNER */}
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
                <div>3. {t('liveStudio.chromeFlagsStep3')}</div>
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
                {errorTitle || t('liveStudio.micErrorTitle')}
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                {micError}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => { setMicError(null); setErrorTitle(null); }}
            className="btn-secondary mobile-touch-target"
            style={{ padding: '0.4rem', borderRadius: '50%', minHeight: '36px', minWidth: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            aria-label={t('liveStudio.dismiss')}
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* TOP HEADER & CONTROLS */}
      <div className="glass-card" style={{ padding: 'clamp(0.75rem, 2vw, 1.25rem)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ flex: '1 1 300px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.3rem', flexWrap: 'wrap' }}>
            <Radio size={22} color="var(--accent-purple)" className={isLiveActive || assistantState !== 'idle' ? "pulse-icon" : ""} />
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800 }}>{t('liveStudio.title')}</h2>
            <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#c084fc', border: '1px solid rgba(168, 85, 247, 0.4)', fontWeight: 700 }}>
              {t('liveStudio.badge')}
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: '1.4' }}>
            {t('liveStudio.subtitle')}
          </p>
        </div>

        {/* Global Action Buttons */}
        <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center', flexWrap: 'wrap' }}>
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
              padding: '0.65rem 1rem',
              minHeight: '44px',
              fontSize: '0.85rem',
              fontWeight: 600,
              borderColor: showSettingsDrawer ? 'var(--accent-cyan)' : 'var(--border-color)',
              background: showSettingsDrawer ? 'rgba(6, 182, 212, 0.15)' : 'rgba(0,0,0,0.25)',
              color: showSettingsDrawer ? 'var(--accent-cyan)' : 'var(--text-main)'
            }}
          >
            <Sliders size={16} />
            <span>{t('liveStudio.liveSettings')}</span>
            {settingsNotice && (
              <span className="badge badge-success" style={{ padding: '0.15rem 0.4rem', fontSize: '0.65rem' }}>
                ✓
              </span>
            )}
          </button>

          {/* Explicit Mute Toggle */}
          <button
            type="button"
            className="btn-secondary mobile-touch-target"
            onClick={handleToggleMute}
            aria-label={isMuted ? (t('liveStudio.unmuteBtn') || 'Riattiva microfono') : (t('liveStudio.muteBtn') || 'Muto')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              padding: '0.65rem 1rem',
              minHeight: '44px',
              fontSize: '0.85rem',
              fontWeight: 600,
              background: isMuted ? 'rgba(239, 68, 68, 0.2)' : 'rgba(0,0,0,0.25)',
              borderColor: isMuted ? '#ef4444' : 'var(--border-color)',
              color: isMuted ? '#fca5a5' : 'var(--text-main)'
            }}
          >
            {isMuted ? <VolumeX size={16} color="#ef4444" /> : <Mic size={16} />}
            <span>{isMuted ? (t('liveStudio.unmuteBtn') || 'Riattiva') : (t('liveStudio.muteBtn') || 'Muto')}</span>
          </button>

          {/* Clear Memory */}
          <button
            type="button"
            className="btn-secondary mobile-touch-target"
            onClick={handleClearMemory}
            aria-label={t('liveStudio.clearMemoryBtn') || 'Azzera memoria'}
            title={t('liveStudio.clearMemoryBtn') || 'Azzera memoria'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              padding: '0.65rem 1rem',
              minHeight: '44px',
              fontSize: '0.85rem',
              fontWeight: 600
            }}
          >
            <Trash2 size={16} />
            <span>{t('liveStudio.clearMemoryBtn') || 'Azzera memoria'}</span>
          </button>

          {/* Mode-specific Primary Start/Stop for Live mode */}
          {mode === 'live' && (
            isLiveActive ? (
              <button
                type="button"
                className="btn-danger mobile-touch-target"
                onClick={handleStopLive}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 1.25rem', minHeight: '44px', minWidth: '44px', fontSize: '0.9rem', fontWeight: 700 }}
              >
                <Square size={16} /> {t('liveStudio.stopSession')}
              </button>
            ) : (
              <button
                type="button"
                className="btn-primary mobile-touch-target"
                onClick={handleStartLive}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 1.25rem', minHeight: '44px', minWidth: '44px', fontSize: '0.9rem', fontWeight: 700, background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}
              >
                <Play size={16} /> {t('liveStudio.startSession')}
              </button>
            )
          )}
        </div>
      </div>

      {/* MODE SEGMENT SWITCHER */}
      <div
        className="glass-card"
        style={{
          padding: '0.5rem',
          display: 'flex',
          gap: '0.5rem',
          borderRadius: '12px',
          background: 'rgba(0,0,0,0.3)',
          flexWrap: 'wrap'
        }}
        role="tablist"
        aria-label="Modalità Assistente Vocale"
      >
        <button
          type="button"
          role="tab"
          id="mode-tab-assistant"
          aria-selected={mode === 'assistant'}
          onClick={() => handleSwitchMode('assistant')}
          className="mobile-touch-target"
          style={{
            flex: '1 1 200px',
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            border: 'none',
            background: mode === 'assistant' ? 'var(--gradient-brand)' : 'transparent',
            color: mode === 'assistant' ? 'white' : 'var(--text-muted)',
            fontWeight: mode === 'assistant' ? 800 : 600,
            fontSize: '0.92rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.6rem',
            minHeight: '48px',
            transition: 'all 0.2s ease'
          }}
        >
          <Bot size={18} />
          <div style={{ textAlign: 'left' }}>
            <div>{t('liveStudio.tabAssistant') || 'Assistente'}</div>
            <div style={{ fontSize: '0.72rem', opacity: 0.8, fontWeight: 400 }}>{t('liveStudio.tabAssistantDesc') || 'Tap-to-Talk • Risposta singola'}</div>
          </div>
        </button>

        <button
          type="button"
          role="tab"
          id="mode-tab-live"
          aria-selected={mode === 'live'}
          onClick={() => handleSwitchMode('live')}
          className="mobile-touch-target"
          style={{
            flex: '1 1 200px',
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            border: 'none',
            background: mode === 'live' ? 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' : 'transparent',
            color: mode === 'live' ? 'white' : 'var(--text-muted)',
            fontWeight: mode === 'live' ? 800 : 600,
            fontSize: '0.92rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.6rem',
            minHeight: '48px',
            transition: 'all 0.2s ease'
          }}
        >
          <Radio size={18} />
          <div style={{ textAlign: 'left' }}>
            <div>{t('liveStudio.tabLive') || 'Live · Sperimentale'}</div>
            <div style={{ fontSize: '0.72rem', opacity: 0.8, fontWeight: 400 }}>{t('liveStudio.tabLiveDesc') || 'Full-duplex continuo con VAD'}</div>
          </div>
        </button>
      </div>

      {/* COMPACT STATUS & PRIVACY TOOLBAR */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.75rem',
          padding: '0.25rem 0.25rem'
        }}
      >
        {/* Real-time Hardware Privacy Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          {connectionStatus === 'connecting' && (
            <span className="badge badge-warning" style={{ fontSize: '0.72rem' }}>
              <span className="pulse-icon">●</span> Connessione...
            </span>
          )}
          {isMuted ? (
            <span className="badge badge-danger" style={{ fontSize: '0.72rem' }}>
              ● {t('liveStudio.micMuted') || 'Microfono mutato'}
            </span>
          ) : isMicPhysicallyActive ? (
            <span className="badge badge-success" style={{ fontSize: '0.72rem' }}>
              <span className="pulse-icon">●</span> {t('liveStudio.micActive') || 'Microfono attivo'}
            </span>
          ) : (
            <span className="badge badge-secondary" style={{ fontSize: '0.72rem' }}>
              🔒 {t('liveStudio.micInactive') || 'Microfono spento'}
            </span>
          )}
          {clearMemoryNotice && (
            <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>
              ✓ {t('liveStudio.clearMemorySuccess') || 'Memoria conversazione azzerata.'}
            </span>
          )}
        </div>

        {/* Toggle Hardware Details Drawer */}
        <button
          type="button"
          onClick={() => setShowHardwareDetails(prev => !prev)}
          className="btn-secondary mobile-touch-target"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.35rem 0.75rem',
            fontSize: '0.8rem',
            borderRadius: '8px',
            minHeight: '38px'
          }}
          aria-expanded={showHardwareDetails}
        >
          {showHardwareDetails ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          <span>{t('liveStudio.hardwareDetailsToggle') || 'Dettagli Hardware & Routing'}</span>
        </button>
      </div>

      {/* LIVE SETTINGS DRAWER */}
      {showSettingsDrawer && (
        <div
          className="glass-card"
          style={{
            padding: '1.25rem 1.5rem',
            borderRadius: '12px',
            border: '1px solid rgba(6, 182, 212, 0.4)',
            background: 'rgba(15, 23, 42, 0.92)',
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
            </div>
            <button
              type="button"
              className="btn-secondary mobile-touch-target"
              onClick={() => setShowSettingsDrawer(false)}
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem', minHeight: '44px' }}
            >
              <X size={15} /> {t('liveStudio.closeSettings')}
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '1.25rem' }}>
            {/* 1. Context Window */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>🧠 {t('liveStudio.contextWindow')}</label>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#38bdf8' }}>{liveSettings.maxContextTurns} turni</span>
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
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>{t('liveStudio.contextWindowDesc')}</span>
            </div>

            {/* 2. Max Tokens */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>📝 {t('liveStudio.maxTokens')}</label>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#a78bfa' }}>{liveSettings.maxTokens} token</span>
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
            </div>

            {/* 3. TTS Mode: Server WAV vs Browser Client */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>🔊 {t('liveStudio.ttsModeLabel') || 'Motore Sintesi Vocale'}</label>
              <select
                value={liveSettings.ttsMode}
                onChange={e => handleUpdateSetting('ttsMode', e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '8px',
                  background: 'rgba(0,0,0,0.4)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-main)',
                  fontSize: '0.85rem',
                  minHeight: '44px'
                }}
              >
                <option value="server">{t('liveStudio.ttsModeServer') || 'Server Kokoro/Audio8 (WAV)'}</option>
                <option value="client">{t('liveStudio.ttsModeClient') || 'Browser Client (SpeechSynthesis)'}</option>
              </select>
              <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                {liveSettings.ttsMode === 'client' ? 'Audio sintetizzato direttamente dal browser senza carico server.' : 'Audio ad alta fedeltà generato dal server Alveare.'}
              </span>
            </div>

            {/* 4. Live Voice Selection */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>🎙️ {t('liveStudio.liveVoice')}</label>
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
                  fontSize: '0.85rem',
                  minHeight: '44px'
                }}
              >
                {isKokoroActive ? (
                  <>
                    <option value="if_sara">if_sara (Italiana Femminile)</option>
                    <option value="im_nicola">im_nicola (Italiano Maschile)</option>
                    <option value="bolt">bolt (Conciso)</option>
                    <option value="valeria">valeria (Ufficiale Alveare)</option>
                    <option value="daino">daino (Italiano Maschile)</option>
                    <option value="af_heart">af_heart (Melodica)</option>
                  </>
                ) : (
                  <>
                    <option value="valeria">valeria (Ufficiale Alveare)</option>
                    <option value="daino">daino (Italiano Maschile)</option>
                    <option value="bolt">bolt (Conciso)</option>
                  </>
                )}
              </select>
            </div>
          </div>
        </div>
      )}

      {/* CENTRAL VOICE CONSOLE */}
      <div className="glass-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '1.25rem', gap: '1.25rem', minHeight: '380px' }}>
        
        {/* Visualizer and Status Display */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '0.5rem 0', gap: '1rem', width: '100%' }}>
          
          <div style={{ position: 'relative', width: '110px', height: '110px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div
              style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                borderRadius: '50%',
                background: (assistantSpeaking || assistantState === 'speaking')
                  ? 'radial-gradient(circle, rgba(16, 185, 129, 0.4) 0%, rgba(16, 185, 129, 0) 70%)'
                  : (userSpeaking || assistantState === 'listening')
                  ? 'radial-gradient(circle, rgba(245, 158, 11, 0.4) 0%, rgba(245, 158, 11, 0) 70%)'
                  : assistantState === 'processing'
                  ? 'radial-gradient(circle, rgba(139, 92, 246, 0.4) 0%, rgba(139, 92, 246, 0) 70%)'
                  : isLiveActive
                  ? 'radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, rgba(168, 85, 247, 0) 70%)'
                  : 'transparent',
                animation: (isLiveActive || assistantState !== 'idle') ? 'pulse 2s infinite' : 'none'
              }}
            />

            {/* Central Action Button */}
            {mode === 'assistant' ? (
              <button
                type="button"
                onClick={handleAssistantTap}
                className="mobile-touch-target"
                aria-label={
                  assistantState === 'idle'
                    ? (t('liveStudio.tapToTalk') || 'Tocca per parlare')
                    : assistantState === 'listening'
                    ? (t('liveStudio.tapToSend') || 'Tocca per inviare')
                    : (t('liveStudio.interruptBtn') || 'Interrompi')
                }
                style={{
                  width: '84px',
                  height: '84px',
                  minWidth: '84px',
                  minHeight: '84px',
                  borderRadius: '50%',
                  border: 'none',
                  background: assistantState === 'listening'
                    ? 'linear-gradient(135deg, #ef4444 0%, #f59e0b 100%)'
                    : assistantState === 'processing'
                    ? 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)'
                    : assistantState === 'speaking'
                    ? 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)'
                    : 'var(--gradient-brand)',
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  boxShadow: assistantState !== 'idle' ? 'var(--shadow-glow)' : 'none',
                  transition: 'all 0.2s ease',
                  zIndex: 2
                }}
              >
                {assistantState === 'listening' ? (
                  <Mic size={36} />
                ) : assistantState === 'processing' ? (
                  <Activity size={34} className="pulse-icon" />
                ) : assistantState === 'speaking' ? (
                  <Volume2 size={36} />
                ) : (
                  <Mic size={36} />
                )}
              </button>
            ) : (
              <button
                type="button"
                onClick={isLiveActive ? handleStopLive : handleStartLive}
                className="mobile-touch-target"
                aria-label={isLiveActive ? t('liveStudio.stopSession') : t('liveStudio.startSession')}
                style={{
                  width: '84px',
                  height: '84px',
                  minWidth: '84px',
                  minHeight: '84px',
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
            )}
          </div>

          {/* Status Label */}
          <div style={{ textAlign: 'center', maxWidth: '500px', width: '100%' }}>
            <div style={{ fontWeight: 700, fontSize: '1.05rem' }}>
              {mode === 'assistant' ? (
                assistantState === 'listening'
                  ? (t('liveStudio.statusListening') || '🎙️ Ti sto ascoltando... (tocca per inviare)')
                  : assistantState === 'processing'
                  ? (t('liveStudio.statusProcessing') || '⏳ Elaborazione risposta in corso...')
                  : assistantState === 'speaking'
                  ? (t('liveStudio.statusSpeaking') || '🔊 Alveare sta parlando...')
                  : (t('liveStudio.tapToTalk') || 'Tocca per parlare')
              ) : (
                assistantSpeaking
                  ? t('liveStudio.statusSpeaking')
                  : userSpeaking
                  ? t('liveStudio.statusListening')
                  : isLiveActive
                  ? t('liveStudio.statusReady')
                  : t('liveStudio.statusInactive')
              )}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem', lineHeight: '1.4' }}>
              {mode === 'assistant' ? (
                assistantState === 'idle'
                  ? 'Fai una domanda: tocca il microfono, parla e ricevi la risposta completa.'
                  : assistantState === 'listening'
                  ? 'Parla liberamente. Tocca il cerchio o attendi la fine della frase.'
                  : assistantState === 'processing'
                  ? 'Generazione della risposta neurale in corso...'
                  : 'Riproduzione audio in corso. Tocca per interrompere.'
              ) : (
                isLiveActive ? t('liveStudio.descActive') : t('liveStudio.descInactive')
              )}
            </div>

            {/* In-turn interrupt shortcut for Assistant mode */}
            {(assistantState === 'speaking' || assistantState === 'processing') && (
              <div style={{ marginTop: '0.6rem' }}>
                <button
                  type="button"
                  className="btn-danger mobile-touch-target"
                  onClick={handleUserInterrupt}
                  style={{ padding: '0.4rem 1rem', fontSize: '0.82rem', minHeight: '44px' }}
                >
                  <StopCircle size={16} /> {t('liveStudio.interruptBtn') || 'Interrompi risposta'}
                </button>
              </div>
            )}
          </div>

          {/* Real-time Canvas Waveform */}
          <canvas
            ref={canvasRef}
            width={320}
            height={55}
            style={{ width: '100%', maxWidth: '340px', height: '55px', borderRadius: '8px', background: 'rgba(0,0,0,0.2)', display: 'block' }}
          />

          {/* Telemetry HUD Toggle */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
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
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', justifyContent: 'center', background: 'rgba(0,0,0,0.3)', padding: '0.75rem 1rem', borderRadius: '12px', border: '1px solid var(--border-color)', width: '100%', maxWidth: '640px' }}>
              <div style={{ textAlign: 'center', minWidth: '85px', flex: '1 1 85px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.sttLatency')}</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#34d399' }}>{sttLatency > 0 ? `${sttLatency} ms` : '—'}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: '95px', flex: '1 1 95px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.ttft')}</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#c084fc' }}>{ttft > 0 ? `${ttft} ms` : '—'}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: '105px', flex: '1 1 105px' }} title={t('liveStudio.serverTtfaTooltip') || 'Latenza primo audio generato dal server'}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.serverTtfaLabel') || 'TTFA Server'}</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#60a5fa' }}>{ttfa > 0 ? `${ttfa} ms` : '—'}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: '105px', flex: '1 1 105px' }} title={t('liveStudio.audibleLatencyTooltip') || 'Latenza effettiva uscita audio nelle casse'}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.audibleLatencyLabel') || 'Latenza Uditiva'}</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#38bdf8' }}>{audibleLatency > 0 ? `${audibleLatency} ms` : '—'}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: '85px', flex: '1 1 85px' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>{t('liveStudio.e2e')}</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#f59e0b' }}>{e2eLatency > 0 ? `${e2eLatency} ms` : '—'}</div>
              </div>
            </div>
          )}
        </div>

        {/* CONVERSATION & TOOL TRANSCRIPT LOG */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '350px', paddingRight: '0.35rem', width: '100%' }}>
          {messages.length === 0 && !streamingAssistantText ? (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem 0', fontSize: '0.9rem' }}>
              {t('liveStudio.emptyTranscript')}
            </div>
          ) : (
            messages.map(msg => {
              if (msg.role === 'tool') {
                return (
                  <div
                    key={msg.id}
                    style={{
                      alignSelf: 'center',
                      width: '90%',
                      maxWidth: '600px',
                      padding: '0.65rem 1rem',
                      borderRadius: '10px',
                      background: 'rgba(15, 23, 42, 0.7)',
                      border: '1px solid rgba(6, 182, 212, 0.3)',
                      color: 'var(--text-main)',
                      fontSize: '0.85rem'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Wrench size={16} color="var(--accent-cyan)" />
                        <span style={{ fontWeight: 700 }}>
                          {msg.status === 'running'
                            ? (t('liveStudio.toolCardRunning', { name: msg.name }) || `Esecuzione strumento: ${msg.name}...`)
                            : (t('liveStudio.toolCardDone', { name: msg.name }) || `Strumento ${msg.name} completato`)}
                        </span>
                      </div>
                      <span className={`badge ${msg.status === 'running' ? 'badge-warning' : msg.status === 'error' ? 'badge-danger' : 'badge-success'}`}>
                        {msg.status}
                      </span>
                    </div>
                    {msg.args && (
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
                        Input: {JSON.stringify(msg.args)}
                      </div>
                    )}
                    {msg.result && (
                      <div style={{ fontSize: '0.78rem', color: '#67e8f9', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
                        Risultato: {String(msg.result)}
                      </div>
                    )}
                  </div>
                );
              }

              return (
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
                    <span>{msg.role === 'user' ? 'Tu' : 'Alveare'}</span>
                    <span>{msg.timestamp}</span>
                  </div>
                  <div>{msg.text}</div>
                  {msg.metrics?.e2e_latency_ms && (
                    <div style={{ fontSize: '0.68rem', opacity: 0.6, marginTop: '0.35rem' }}>
                      ⚡ E2E: {msg.metrics.e2e_latency_ms} ms (STT: {msg.metrics.stt_latency_ms} ms | TTFT: {msg.metrics.ttft_ms} ms)
                    </div>
                  )}
                </div>
              );
            })
          )}

          {/* Real-time Streaming Tokens (Deduped upon turn_complete) */}
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
              <div style={{ fontSize: '0.7rem', opacity: 0.7, marginBottom: '0.25rem' }}>Alveare (Generazione...)</div>
              <div>{streamingAssistantText}</div>
            </div>
          )}
        </div>

      </div>

      {/* COLLAPSIBLE HARDWARE SLOTS & ROUTING DETAILS */}
      <div
        id="hardware-details-section"
        style={{
          maxHeight: showHardwareDetails ? '2000px' : '0px',
          overflow: 'hidden',
          transition: 'max-height 0.35s ease',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem'
        }}
      >
        {/* WAKEWORD & PRIVACY NOTICE */}
        <div
          style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid var(--border-color)',
            borderRadius: '10px',
            padding: '0.65rem 1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.8rem',
            color: 'var(--text-muted)'
          }}
        >
          <ShieldAlert size={16} color="var(--accent-cyan)" style={{ flexShrink: 0 }} />
          <span>
            {t('liveStudio.wakewordNotice') || 'Wakeword locale: gestita in hardware dall’app nativa Android Alveare. Nel browser web: tocca per parlare (Assistente) o avvia la sessione continua (Live).'}
          </span>
        </div>

        {/* ULTRA-FAST LIVE BANNER / TOGGLE */}
        <div
          className="glass-card"
          style={{
            padding: '0.75rem 1.25rem',
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
              gap: '0.35rem',
              minHeight: '44px'
            }}
            aria-label={ultraFastLive ? t('liveStudio.ultraFastToggleOn') : t('liveStudio.ultraFastToggleOff')}
          >
            <Zap size={13} />
            <span>{ultraFastLive ? t('liveStudio.ultraFastToggleOn') : t('liveStudio.ultraFastToggleOff')}</span>
          </button>
        </div>

        {/* COMPACT HARDWARE SLOTS ROUTING CARDS */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '1rem' }}>
          {/* GPU: LLM Chat */}
          <div className="glass-card" style={{ padding: '0.85rem 1.15rem', borderLeft: '4px solid #c084fc', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#c084fc', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                🎮 {t('liveStudio.gpuCardTitle')}
              </span>
              <span className="badge badge-success" style={{ fontSize: '0.68rem' }}>{t('liveStudio.gpuCardRole')}</span>
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 800, wordBreak: 'break-all' }}>{llmSlot.model || 'gemma4-e4b'}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              {t('liveStudio.gpuCardDesc', { load: status?.gpu_usage?.percent || 0, tps: status?.tok_per_sec || 35 })}
            </div>
          </div>

          {/* NPU / CPU: Kokoro / Audio8 TTS */}
          <div className="glass-card" style={{ padding: '0.85rem 1.15rem', borderLeft: '4px solid #34d399', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#34d399', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                ⚡ {t('liveStudio.npuCardTitle')}
              </span>
              <span className="badge badge-success" style={{ fontSize: '0.68rem' }}>
                {ttsSlot.device === 'npu' ? t('liveStudio.npuCardRole') : (sttSlot.device === 'npu' ? t('liveStudio.cpuCardRole') : t('liveStudio.npuCardRole'))}
              </span>
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 800, wordBreak: 'break-all' }}>
              {ttsSlot.model || 'hexgrad/Kokoro-82M'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              {t('liveStudio.npuCardDesc')}
            </div>
          </div>

          {/* CPU: Whisper STT */}
          <div className="glass-card" style={{ padding: '0.85rem 1.15rem', borderLeft: '4px solid #60a5fa', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                🖥️ {t('liveStudio.cpuCardTitle')}
              </span>
              <span className="badge badge-success" style={{ fontSize: '0.68rem' }}>
                {sttSlot.device === 'cpu' ? t('liveStudio.cpuCardRole') : (ttsSlot.device === 'cpu' ? t('liveStudio.npuCardRole') : t('liveStudio.cpuCardRole'))}
              </span>
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 800, wordBreak: 'break-all' }}>
              {sttSlot.model || 'openai/whisper-large-v3-turbo'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              {t('liveStudio.cpuCardDesc')}
            </div>
          </div>
        </div>

        {onNavigateToControl && (
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '-0.25rem' }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={onNavigateToControl}
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.65rem' }}
            >
              ⚙️ Control Panel Slot
            </button>
          </div>
        )}
      </div>

    </div>
  );
}
