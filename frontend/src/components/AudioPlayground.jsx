import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Mic,
  MicOff,
  Upload,
  Play,
  Pause,
  Copy,
  Check,
  Trash2,
  Download,
  Volume2,
  Sparkles,
  Zap,
  Activity,
  Music,
  FileAudio,
  AlertCircle,
  Clock,
  Radio,
  FileText,
  Globe,
  Sliders,
  Server
} from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';

export default function AudioPlayground({ apiBase, status, activeModel, isServerRunning, models = [], onNavigateToControl }) {
  const { t } = useTranslation();

  // Mode: 'realtime' (Live mic stream) | 'file' (Audio upload)
  const [activeMode, setActiveMode] = useState('realtime');
  
  // Language selection: 'auto' | 'it' | 'en' | 'zh' | 'ja' | 'ko' | 'yue'
  const [selectedLanguage, setSelectedLanguage] = useState(() => {
    return localStorage.getItem('alveare_stt_language') || 'auto';
  });

  const handleLanguageChange = (lang) => {
    setSelectedLanguage(lang);
    localStorage.setItem('alveare_stt_language', lang);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "set_language", language: lang }));
    }
  };

  // Real-time streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const isStreamingRef = useRef(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [streamHistory, setStreamHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('alveare_stt_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('alveare_stt_history', JSON.stringify(streamHistory));
    } catch {}
  }, [streamHistory]);

  const [streamStats, setStreamStats] = useState({
    language: 'auto',
    latency_ms: 0
  });

  // File upload state
  const [uploadedFile, setUploadedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isTranscribingFile, setIsTranscribingFile] = useState(false);
  const [fileTranscript, setFileTranscript] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  // UI state
  const [copied, setCopied] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Refs
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const audioElementRef = useRef(null);
  const fileInputRef = useRef(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopRealtimeStream();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  // Visualizer drawing function
  const drawWaveform = useCallback((analyser) => {
    if (!canvasRef.current || !analyser) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const renderFrame = () => {
      animFrameRef.current = requestAnimationFrame(renderFrame);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height * 0.85;

        // Gradient color from Cyan to Purple
        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
        gradient.addColorStop(0, '#06b6d4');
        gradient.addColorStop(1, '#8b5cf6');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth + 1;
      }
    };

    renderFrame();
  }, []);

function downsampleBuffer(buffer, sampleRate, outSampleRate = 16000) {
  if (outSampleRate >= sampleRate) {
    return buffer;
  }
  const sampleRateRatio = sampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

  // Real-time stream starter
  const startRealtimeStream = async () => {
    if (!isServerRunning) {
      setErrorMsg(t('audioPlayground.serverOffBannerDesc'));
      return;
    }
    setErrorMsg('');
    setIsFinalizing(false);
    try {
      // 1. Get user microphone stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      mediaStreamRef.current = stream;

      // 2. Setup AudioContext
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }
      audioContextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 128;
      source.connect(analyser);
      drawWaveform(analyser);

      // 3. Connect WebSocket to /ws/stt
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = apiBase.replace(/^https?:\/\//, '');
      const wsUrl = `${wsProtocol}//${wsHost}/ws/stt`;

      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setIsStreaming(true);
        isStreamingRef.current = true;
        if (selectedLanguage !== 'auto') {
          ws.send(JSON.stringify({ action: "set_language", language: selectedLanguage }));
        }
      };

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'partial' || data.type === 'final') {
            if (data.text !== undefined && data.text !== null && data.text.trim()) {
              setLiveTranscript(data.text);
            }
            setStreamStats({
              language: data.language || 'auto',
              latency_ms: data.latency_ms || 0
            });
            if (data.is_final) {
              setIsFinalizing(false);
              if (data.text && data.text.trim()) {
                const newEntry = {
                  id: Date.now() + '-' + Math.random().toString(36).substring(2, 7),
                  text: data.text.trim(),
                  time: new Date().toLocaleTimeString(),
                  language: data.language || 'auto',
                  latency_ms: data.latency_ms || 0
                };
                setStreamHistory(prev => [newEntry, ...prev]);
              }
              if (!isStreamingRef.current) {
                try { ws.close(); } catch (e) {}
                wsRef.current = null;
              }
            }
          }
        } catch (e) {
          console.error("WS message parse error:", e);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        setErrorMsg('Errore di connessione al WebSocket STT.');
        setIsFinalizing(false);
      };

      ws.onclose = () => {
        setIsStreaming(false);
        isStreamingRef.current = false;
        setIsFinalizing(false);
      };

      // 4. Create ScriptProcessorNode to capture and downsample to 16kHz PCM
      const bufferSize = 4096;
      const processor = ctx.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN && isStreamingRef.current) {
          const inputData = e.inputBuffer.getChannelData(0);
          const downsampled = downsampleBuffer(inputData, ctx.sampleRate, 16000);
          // Convert float32 [-1.0, 1.0] to int16 PCM [-32768, 32767]
          const int16Buffer = new Int16Array(downsampled.length);
          for (let i = 0; i < downsampled.length; i++) {
            const s = Math.max(-1, Math.min(1, downsampled[i]));
            int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          ws.send(int16Buffer.buffer);
        }
      };

      const muteGain = ctx.createGain();
      muteGain.gain.value = 0;
      source.connect(processor);
      processor.connect(muteGain);
      muteGain.connect(ctx.destination);

    } catch (err) {
      console.error("Microphone access error:", err);
      setErrorMsg(t('audioPlayground.micError', { error: err.message }));
      stopRealtimeStream();
    }
  };

  const stopRealtimeStream = () => {
    isStreamingRef.current = false;
    setIsStreaming(false);

    if (processorRef.current) {
      try { processorRef.current.disconnect(); } catch (e) {}
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (e) {}
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      try { mediaStreamRef.current.getTracks().forEach(t => t.stop()); } catch (e) {}
      mediaStreamRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setIsFinalizing(true);
      wsRef.current.send(JSON.stringify({ action: "flush" }));
      setTimeout(() => {
        if (wsRef.current) {
          try { wsRef.current.close(); } catch (e) {}
          wsRef.current = null;
        }
        setIsFinalizing(false);
      }, 10000);
    } else {
      setIsFinalizing(false);
    }
  };

  // Copy helper
  const handleCopy = (text) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Export helper
  const handleExport = (text, filename = "transcription.txt") => {
    if (!text) return;
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // File Upload Handlers
  const handleFileSelect = async (file) => {
    if (!file) return;
    if (!isServerRunning) {
      setErrorMsg(t('audioPlayground.serverOffBannerDesc'));
      return;
    }
    setErrorMsg('');
    setUploadedFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setFileTranscript(null);

    // Automatically trigger transcription
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', activeModel || 'whisper-base');
    if (selectedLanguage !== 'auto') {
      formData.append('language', selectedLanguage);
    }
    formData.append('response_format', 'verbose_json');

    setIsTranscribingFile(true);
    const t0 = performance.now();
    try {
      const res = await fetch(`${apiBase}/v1/audio/transcriptions`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
      const data = await res.json();
      const elapsed = Math.round(performance.now() - t0);
      setFileTranscript({
        ...data,
        clientElapsedMs: elapsed
      });
    } catch (err) {
      console.error("Audio file transcription error:", err);
      setErrorMsg(t('audioPlayground.micError', { error: err.message }));
    } finally {
      setIsTranscribingFile(false);
    }
  };

  const getLanguageLabel = (lang) => {
    switch ((lang || '').toLowerCase()) {
      case 'it': return '🇮🇹 Italiano (IT)';
      case 'en': return '🇬🇧 English (EN)';
      case 'es': return '🇪🇸 Español (ES)';
      case 'fr': return '🇫🇷 Français (FR)';
      case 'de': return '🇩🇪 Deutsch (DE)';
      case 'pt': return '🇵🇹 Português (PT)';
      case 'zh': return '🇨🇳 中文 (ZH)';
      case 'ja': return '🇯🇵 日本語 (JA)';
      case 'ko': return '🇰🇷 한국어 (KO)';
      case 'ru': return '🇷🇺 Русский (RU)';
      case 'ar': return '🇸🇦 العربية (AR)';
      case 'nl': return '🇳🇱 Nederlands (NL)';
      case 'pl': return '🇵🇱 Polski (PL)';
      case 'tr': return '🇹🇷 Türkçe (TR)';
      case 'auto': return '🌐 Auto-Detect';
      default: return `🌐 ${lang ? lang.toUpperCase() : 'Auto'}`;
    }
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)', overflow: 'hidden', background: 'var(--bg-primary)', position: 'relative' }}>
      
      {/* Header Banner */}
      <div style={{
        padding: '1.25rem 2rem',
        borderBottom: '1px solid var(--border-color)',
        background: 'linear-gradient(90deg, rgba(6, 182, 212, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 15px rgba(6, 182, 212, 0.3)'
          }}>
            <Radio size={24} color="white" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 800, margin: 0, color: 'var(--text-main)' }}>
                {t('audioPlayground.title')}
              </h2>
              {isServerRunning ? (
                <span className="badge badge-success" style={{ fontSize: '0.72rem' }}>
                  <span className="pulse-icon">●</span> {t('audioPlayground.liveAudioBadge')}
                </span>
              ) : (
                <span className="badge badge-danger" style={{ fontSize: '0.72rem' }}>
                  ● {t('nav.serverStopped')}
                </span>
              )}
            </div>
            <p style={{ margin: '0.2rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              {t('audioPlayground.subtitle')}
            </p>
          </div>
        </div>

        {/* Controls Bar: Language Selector & Mode Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', flexWrap: 'wrap' }}>
          
          {/* Explicit Language Dropdown Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'var(--bg-secondary)', padding: '0.3rem 0.65rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <Globe size={15} color="var(--accent-cyan)" />
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>
              {t('audioPlayground.languageSelectLabel')}
            </span>
            <select
              value={selectedLanguage}
              onChange={(e) => handleLanguageChange(e.target.value)}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-main)',
                fontSize: '0.82rem',
                fontWeight: 700,
                cursor: 'pointer',
                outline: 'none'
              }}
            >
              <option value="auto" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🌐 Auto (Rilevamento)</option>
              <option value="it" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇮🇹 Italiano</option>
              <option value="en" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇬🇧 English</option>
              <option value="es" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇪🇸 Español</option>
              <option value="fr" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇫🇷 Français</option>
              <option value="de" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇩🇪 Deutsch</option>
              <option value="pt" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇵🇹 Português</option>
              <option value="zh" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇨🇳 中文</option>
              <option value="ja" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇯🇵 日本語</option>
              <option value="ko" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇰🇷 한국어</option>
              <option value="ru" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇷🇺 Русский</option>
              <option value="ar" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇸🇦 العربية</option>
              <option value="nl" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇳🇱 Nederlands</option>
              <option value="pl" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇵🇱 Polski</option>
              <option value="tr" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇹🇷 Türkçe</option>
            </select>
          </div>

          {/* Mode Switcher Tabs */}
          <div style={{ display: 'flex', background: 'var(--bg-secondary)', padding: '0.3rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <button
              onClick={() => { setActiveMode('realtime'); }}
              style={{
                padding: '0.45rem 0.85rem',
                borderRadius: '7px',
                border: 'none',
                background: activeMode === 'realtime' ? 'var(--gradient-brand)' : 'transparent',
                color: activeMode === 'realtime' ? '#fff' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: '0.82rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.45rem',
                transition: 'all 0.2s ease'
              }}
            >
              <Mic size={15} /> {t('audioPlayground.liveStreamTab')}
            </button>
            <button
              onClick={() => { setActiveMode('file'); stopRealtimeStream(); }}
              style={{
                padding: '0.45rem 0.85rem',
                borderRadius: '7px',
                border: 'none',
                background: activeMode === 'file' ? 'var(--gradient-brand)' : 'transparent',
                color: activeMode === 'file' ? '#fff' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: '0.82rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.45rem',
                transition: 'all 0.2s ease'
              }}
            >
              <FileAudio size={15} /> {t('audioPlayground.fileTab')}
            </button>
          </div>
        </div>
      </div>

      {/* Server Stopped Grayed-out Disabled Banner */}
      {!isServerRunning && (
        <div style={{
          padding: '1rem 2rem',
          background: 'linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(245, 158, 11, 0.12) 100%)',
          borderBottom: '1px solid rgba(239, 68, 68, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          zIndex: 20
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <AlertCircle size={20} color="#ef4444" />
            <div>
              <strong style={{ color: 'var(--text-main)', fontSize: '0.92rem' }}>
                {t('audioPlayground.serverOffBannerTitle')}
              </strong>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginTop: '0.1rem' }}>
                {t('audioPlayground.serverOffBannerDesc')}
              </div>
            </div>
          </div>
          {onNavigateToControl && (
            <button
              className="btn-primary"
              onClick={onNavigateToControl}
              style={{ padding: '0.45rem 1rem', fontSize: '0.82rem', borderRadius: '8px' }}
            >
              <Server size={14} /> {t('audioPlayground.goToControlPanel')}
            </button>
          )}
        </div>
      )}

      {/* Error notification */}
      {errorMsg && (
        <div style={{ padding: '0.75rem 2rem', background: 'rgba(239, 68, 68, 0.15)', borderBottom: '1px solid rgba(239, 68, 68, 0.3)', display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#fca5a5', fontSize: '0.88rem' }}>
          <AlertCircle size={18} color="#ef4444" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Body with Disabled Dimming when server is stopped */}
      <div style={{
        flex: 1,
        display: 'flex',
        overflow: 'hidden',
        padding: '1.5rem 2rem',
        gap: '1.5rem',
        opacity: isServerRunning ? 1 : 0.45,
        pointerEvents: isServerRunning ? 'auto' : 'none',
        filter: isServerRunning ? 'none' : 'grayscale(0.6)'
      }}>
        
        {/* Left Interactive Control Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1.25rem', overflowY: 'auto' }}>
          
          {activeMode === 'realtime' ? (
            /* REALTIME STREAMING MODE */
            <div style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '16px',
              padding: '1.75rem',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              minHeight: '320px',
              boxShadow: '0 8px 30px rgba(0,0,0,0.15)'
            }}>
              
              {/* Canvas Visualizer */}
              <canvas
                ref={canvasRef}
                width={480}
                height={90}
                style={{
                  width: '100%',
                  maxWidth: '480px',
                  height: '90px',
                  marginBottom: '1rem',
                  borderRadius: '10px',
                  background: 'rgba(0,0,0,0.2)'
                }}
              />

              {/* Central Mic Button */}
              <button
                onClick={isStreaming ? stopRealtimeStream : startRealtimeStream}
                disabled={!isServerRunning || isFinalizing}
                style={{
                  width: '84px',
                  height: '84px',
                  borderRadius: '50%',
                  border: 'none',
                  background: isStreaming
                    ? 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
                    : isFinalizing
                    ? 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                    : 'linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%)',
                  color: 'white',
                  cursor: (isServerRunning && !isFinalizing) ? 'pointer' : 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: isStreaming
                    ? '0 0 35px rgba(239, 68, 68, 0.6)'
                    : isFinalizing
                    ? '0 0 30px rgba(245, 158, 11, 0.5)'
                    : '0 0 30px rgba(6, 182, 212, 0.4)',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
                }}
                className={isStreaming || isFinalizing ? 'pulse-icon' : ''}
                title={isStreaming ? t('audioPlayground.stopListening') : t('audioPlayground.startListening')}
              >
                {isStreaming ? <MicOff size={36} /> : isFinalizing ? <Activity size={36} /> : <Mic size={36} />}
              </button>

              <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                <span style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-main)' }}>
                  {isFinalizing
                    ? '⏳ ' + (t('audioPlayground.finalizing') || 'Elaborazione trascrizione finale in corso...')
                    : isStreaming
                    ? t('audioPlayground.listeningPrompt')
                    : t('audioPlayground.clickToListen')}
                </span>
                <p style={{ margin: '0.25rem 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  {isStreaming ? t('audioPlayground.streamingSubtext') : t('audioPlayground.zeroDelaySubtext')}
                </p>
              </div>

              {/* Status Indicator Bar */}
              {(isStreaming || isFinalizing) && (
                <div style={{
                  display: 'flex',
                  gap: '0.85rem',
                  marginTop: '1.25rem',
                  background: 'rgba(0,0,0,0.3)',
                  padding: '0.5rem 1rem',
                  borderRadius: '20px',
                  border: '1px solid var(--border-color)',
                  flexWrap: 'wrap',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}>
                  <span style={{ fontSize: '0.78rem', color: 'var(--accent-cyan)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Activity size={13} /> {t('audioPlayground.latency', { latency: streamStats.latency_ms > 0 ? streamStats.latency_ms : '<300' })}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: '#10b981', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Globe size={13} /> {getLanguageLabel(streamStats.language)}
                  </span>
                </div>
              )}

            </div>
          ) : (
            /* FILE UPLOAD MODE */
            <div style={{
              background: 'var(--bg-secondary)',
              border: `2px dashed ${dragOver ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
              borderRadius: '16px',
              padding: '2rem',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              minHeight: '320px',
              cursor: isServerRunning ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s ease'
            }}
            onDragOver={(e) => { e.preventDefault(); if (isServerRunning) setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (isServerRunning && e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0]);
            }}
            onClick={() => isServerRunning && fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,.mp3,.wav,.ogg,.m4a,.flac,.webm"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    handleFileSelect(e.target.files[0]);
                  }
                  e.target.value = '';
                }}
              />

              <div style={{
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                background: 'rgba(6, 182, 212, 0.12)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1rem'
              }}>
                <Upload size={32} color="var(--accent-cyan)" />
              </div>

              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-main)' }}>
                {uploadedFile ? uploadedFile.name : t('audioPlayground.dropFilePrompt')}
              </span>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                {t('audioPlayground.supportedFormats')}
              </p>

              {uploadedFile && (
                <div style={{ marginTop: '1.5rem', width: '100%', maxWidth: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }} onClick={(e) => e.stopPropagation()}>
                  <audio
                    ref={audioElementRef}
                    src={audioUrl}
                    controls
                    style={{ width: '100%', height: '40px', borderRadius: '8px' }}
                  />
                  <div style={{ display: 'flex', gap: '0.6rem' }}>
                    <button
                      className="btn-secondary"
                      style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload size={14} /> {t('audioPlayground.replaceAudio')}
                    </button>
                    <button
                      className="btn-secondary"
                      style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#f87171' }}
                      onClick={() => {
                        setUploadedFile(null);
                        setAudioUrl(null);
                        setFileTranscript(null);
                      }}
                    >
                      <Trash2 size={14} /> {t('audioPlayground.removeAudio')}
                    </button>
                  </div>
                </div>
              )}

              {isTranscribingFile && (
                <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--accent-cyan)', fontSize: '0.9rem', fontWeight: 600 }}>
                  <div className="pulse-icon" style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--accent-cyan)' }} />
                  {t('audioPlayground.transcribingFile')}
                </div>
              )}

            </div>
          )}

          {/* Quick Realtime Live Transcript Box */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '14px',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <Sparkles size={16} color="var(--accent-cyan)" />
                {activeMode === 'realtime' ? t('audioPlayground.liveTranscriptTitle') : t('audioPlayground.fileTranscriptTitle')}
              </span>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className="btn-secondary"
                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
                  onClick={() => handleCopy(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                  disabled={!(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                >
                  {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                  {copied ? t('audioPlayground.copied') : t('audioPlayground.copy')}
                </button>
                <button
                  className="btn-secondary"
                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
                  onClick={() => handleExport(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                  disabled={!(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                >
                  <Download size={14} /> {t('audioPlayground.export')}
                </button>
                <button
                  className="btn-secondary"
                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}
                  onClick={() => { setLiveTranscript(''); setFileTranscript(null); }}
                  title={t('audioPlayground.clear')}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>

            <div style={{
              minHeight: '90px',
              maxHeight: '180px',
              overflowY: 'auto',
              padding: '0.85rem',
              background: 'rgba(0,0,0,0.25)',
              borderRadius: '10px',
              border: '1px solid rgba(255,255,255,0.05)',
              fontSize: '1rem',
              lineHeight: 1.6,
              color: (activeMode === 'realtime' ? liveTranscript : fileTranscript?.text) ? 'var(--text-main)' : 'var(--text-muted)',
              fontStyle: (activeMode === 'realtime' ? liveTranscript : fileTranscript?.text) ? 'normal' : 'italic'
            }}>
              {activeMode === 'realtime'
                ? (liveTranscript || (isFinalizing ? ('⏳ ' + (t('audioPlayground.finalizing') || 'Elaborazione trascrizione finale in corso...')) : isStreaming ? t('audioPlayground.waitingForAudio') : t('audioPlayground.pressMicToSpeak')))
                : (fileTranscript?.text || (isTranscribingFile ? t('audioPlayground.transcribingFile') : t('audioPlayground.uploadFileToTranscribe')))}
            </div>

            {/* File metadata card */}
            {activeMode === 'file' && fileTranscript && (
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', paddingTop: '0.25rem' }}>
                <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>
                  {getLanguageLabel(fileTranscript.language)}
                </span>
                <span className="badge badge-secondary" style={{ fontSize: '0.75rem' }}>
                  ⏱️ {fileTranscript.latency_ms} ms latenza
                </span>
              </div>
            )}

          </div>

        </div>

        {/* Right Session History Panel */}
        <div style={{
          width: '340px',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '16px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <div style={{
            padding: '1rem 1.25rem',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.5rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.92rem', color: 'var(--text-main)' }}>
              <Clock size={16} color="var(--accent-purple)" />
              <span>{t('audioPlayground.historyTitle')}</span>
              {streamHistory.length > 0 && (
                <span style={{ fontSize: '0.72rem', background: 'rgba(255,255,255,0.1)', padding: '0.1rem 0.45rem', borderRadius: '10px', color: 'var(--text-muted)' }}>
                  {streamHistory.length}
                </span>
              )}
            </div>
            {streamHistory.length > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <button
                  onClick={() => {
                    const allText = streamHistory.map(h => `[${h.time}] ${h.text}`).join('\n');
                    handleCopy(allText);
                  }}
                  style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                  title="Copia tutto lo storico"
                >
                  <Copy size={13} />
                  <span>Copia Tutto</span>
                </button>
                <button
                  onClick={() => {
                    setStreamHistory([]);
                    try { localStorage.removeItem('alveare_stt_history'); } catch {}
                  }}
                  style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                  title="Svuota cronologia"
                >
                  <Trash2 size={13} />
                  <span>{t('audioPlayground.clear')}</span>
                </button>
              </div>
            )}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {streamHistory.length === 0 ? (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', padding: '2rem 1rem' }}>
                {t('audioPlayground.emptyHistory')}
              </div>
            ) : (
              streamHistory.map((item, idx) => (
                <div
                  key={item.id || idx}
                  style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '10px',
                    padding: '0.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                    transition: 'border-color 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <span style={{ fontWeight: 600 }}>{item.time}</span>
                      {item.language && (
                        <span style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', padding: '0.1rem 0.35rem', borderRadius: '4px', fontSize: '0.65rem', fontWeight: 600 }}>
                          {getLanguageLabel(item.language)}
                        </span>
                      )}
                      {item.latency_ms > 0 && (
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                          {Math.round(item.latency_ms)}ms
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <button
                        onClick={() => handleCopy(item.text)}
                        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '2px' }}
                        title="Copia frase"
                      >
                        <Copy size={12} />
                      </button>
                      <button
                        onClick={() => setStreamHistory(prev => prev.filter((_, i) => i !== idx))}
                        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '2px' }}
                        title="Rimuovi"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.88rem', color: 'var(--text-main)', lineHeight: 1.4, wordBreak: 'break-word' }}>
                    {item.text}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>

    </div>
  );
}
