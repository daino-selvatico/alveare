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
  FileText
} from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';

export default function AudioPlayground({ apiBase, status, activeModel, isServerRunning, models = [] }) {
  const { t } = useTranslation();

  // Mode: 'realtime' (Live mic stream) | 'file' (Audio upload)
  const [activeMode, setActiveMode] = useState('realtime');
  
  // Real-time streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [streamHistory, setStreamHistory] = useState([]);
  const [streamStats, setStreamStats] = useState({
    language: 'it',
    emotion: 'NEUTRAL',
    event: 'Speech',
    latency_ms: 0
  });

  // File upload state
  const [uploadedFile, setUploadedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
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

  // Real-time stream starter
  const startRealtimeStream = async () => {
    setErrorMsg('');
    try {
      // 1. Get user microphone stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      mediaStreamRef.current = stream;

      // 2. Setup AudioContext and downsampling to 16kHz
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx({ sampleRate: 16000 });
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
      };

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.text) {
            setLiveTranscript(data.text);
            setStreamStats({
              language: data.language || 'auto',
              emotion: data.emotion || 'NEUTRAL',
              event: data.event || 'Speech',
              latency_ms: data.latency_ms || 0
            });
            if (data.is_final) {
              setStreamHistory(prev => [...prev, {
                text: data.text,
                time: new Date().toLocaleTimeString(),
                emotion: data.emotion,
                event: data.event,
                language: data.language
              }]);
            }
          }
        } catch (e) {
          console.error("WS message parse error:", e);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        setErrorMsg('Errore di connessione al WebSocket STT.');
      };

      ws.onclose = () => {
        setIsStreaming(false);
      };

      // 4. Create ScriptProcessorNode to capture PCM chunks
      const bufferSize = 4096;
      const processor = ctx.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          // Convert float32 [-1.0, 1.0] to int16 PCM [-32768, 32767]
          const int16Buffer = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          ws.send(int16Buffer.buffer);
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);

    } catch (err) {
      console.error("Microphone access error:", err);
      setErrorMsg(`Impossibile accedere al microfono: ${err.message}`);
      stopRealtimeStream();
    }
  };

  const stopRealtimeStream = () => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "flush" }));
      }
      setTimeout(() => {
        try { wsRef.current.close(); } catch (e) {}
        wsRef.current = null;
      }, 200);
    }
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
    }
    setIsStreaming(false);
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
    setErrorMsg('');
    setUploadedFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setFileTranscript(null);

    // Automatically trigger transcription
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', 'sensevoice');
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
      setErrorMsg(`Errore durante la trascrizione del file: ${err.message}`);
    } finally {
      setIsTranscribingFile(false);
    }
  };

  const getEmotionBadge = (emo) => {
    switch ((emo || '').toUpperCase()) {
      case 'HAPPY': return { label: '😄 Felice / Happy', color: '#10b981' };
      case 'SAD': return { label: '😢 Triste / Sad', color: '#3b82f6' };
      case 'ANGRY': return { label: '😡 Arrabbiato / Angry', color: '#ef4444' };
      case 'FEARFUL': return { label: '😨 Timoroso', color: '#8b5cf6' };
      case 'SURPRISED': return { label: '😲 Sorpreso', color: '#f59e0b' };
      default: return { label: '😐 Neutro / Standard', color: 'var(--text-muted)' };
    }
  };

  const getLanguageLabel = (lang) => {
    switch ((lang || '').toLowerCase()) {
      case 'it': return '🇮🇹 Italiano';
      case 'en': return '🇬🇧 English';
      case 'zh': return '🇨🇳 Cinese';
      case 'ja': return '🇯🇵 Giapponese';
      case 'ko': return '🇰🇷 Coreano';
      case 'yue': return '🇭🇰 Cantonese';
      default: return `🌐 ${lang || 'Auto'}`;
    }
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)', overflow: 'hidden', background: 'var(--bg-primary)' }}>
      
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
                SenseVoice Small STT Workbench
              </h2>
              <span className="badge badge-success" style={{ fontSize: '0.72rem' }}>
                <span className="pulse-icon">●</span> Audio Real-Time
              </span>
            </div>
            <p style={{ margin: '0.2rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Trascrizione vocale istantanea (&lt;30ms) non-autoregressiva con riconoscimento emozioni, eventi e multilingua.
            </p>
          </div>
        </div>

        {/* Mode Switcher Tabs */}
        <div style={{ display: 'flex', background: 'var(--bg-secondary)', padding: '0.3rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          <button
            onClick={() => { setActiveMode('realtime'); }}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '7px',
              border: 'none',
              background: activeMode === 'realtime' ? 'var(--gradient-brand)' : 'transparent',
              color: activeMode === 'realtime' ? '#fff' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '0.84rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease'
            }}
          >
            <Mic size={16} /> Streaming Live (Microfono)
          </button>
          <button
            onClick={() => { setActiveMode('file'); stopRealtimeStream(); }}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '7px',
              border: 'none',
              background: activeMode === 'file' ? 'var(--gradient-brand)' : 'transparent',
              color: activeMode === 'file' ? '#fff' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '0.84rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s ease'
            }}
          >
            <FileAudio size={16} /> File Audio
          </button>
        </div>
      </div>

      {/* Error notification */}
      {errorMsg && (
        <div style={{ padding: '0.75rem 2rem', background: 'rgba(239, 68, 68, 0.15)', borderBottom: '1px solid rgba(239, 68, 68, 0.3)', display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#fca5a5', fontSize: '0.88rem' }}>
          <AlertCircle size={18} color="#ef4444" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', padding: '1.5rem 2rem', gap: '1.5rem' }}>
        
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
                style={{
                  width: '84px',
                  height: '84px',
                  borderRadius: '50%',
                  border: 'none',
                  background: isStreaming
                    ? 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
                    : 'linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%)',
                  color: 'white',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: isStreaming
                    ? '0 0 35px rgba(239, 68, 68, 0.6)'
                    : '0 0 30px rgba(6, 182, 212, 0.4)',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
                }}
                className={isStreaming ? 'pulse-icon' : ''}
                title={isStreaming ? "Interrompi ascolto" : "Avvia ascolto real-time"}
              >
                {isStreaming ? <MicOff size={36} /> : <Mic size={36} />}
              </button>

              <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                <span style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-main)' }}>
                  {isStreaming ? 'In ascolto continuo... Parla liberamente' : 'Clicca per avviare la trascrizione vocale in streaming'}
                </span>
                <p style={{ margin: '0.25rem 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  {isStreaming ? 'Streaming continuo via WebSocket su /ws/stt a 16kHz' : 'Nessun ritardo: le parole appaiono in tempo reale'}
                </p>
              </div>

              {/* Status Indicator Bar */}
              {isStreaming && (
                <div style={{
                  display: 'flex',
                  gap: '0.85rem',
                  marginTop: '1.25rem',
                  background: 'rgba(0,0,0,0.3)',
                  padding: '0.5rem 1rem',
                  borderRadius: '20px',
                  border: '1px solid var(--border-color)',
                  flexWrap: 'wrap',
                  justifyContent: 'center'
                }}>
                  <span style={{ fontSize: '0.78rem', color: 'var(--accent-cyan)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Activity size={13} /> {streamStats.latency_ms > 0 ? `${streamStats.latency_ms} ms` : '&lt;30 ms'} latenza
                  </span>
                  <span style={{ fontSize: '0.78rem', color: '#10b981', fontWeight: 600 }}>
                    {getLanguageLabel(streamStats.language)}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: getEmotionBadge(streamStats.emotion).color, fontWeight: 600 }}>
                    {getEmotionBadge(streamStats.emotion).label}
                  </span>
                  {streamStats.event && streamStats.event !== 'Speech' && (
                    <span style={{ fontSize: '0.78rem', color: 'var(--accent-purple)', fontWeight: 600 }}>
                      🎵 {streamStats.event}
                    </span>
                  )}
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
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0]);
            }}
            onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,.mp3,.wav,.ogg,.m4a,.flac"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
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
                {uploadedFile ? uploadedFile.name : 'Trascina o carica un file audio'}
              </span>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Supporta MP3, WAV, M4A, OGG, FLAC (elaborazione ultrarapida &lt;100ms)
              </p>

              {uploadedFile && (
                <div style={{ marginTop: '1.5rem', width: '100%', maxWidth: '400px' }} onClick={(e) => e.stopPropagation()}>
                  <audio
                    ref={audioElementRef}
                    src={audioUrl}
                    controls
                    style={{ width: '100%', height: '40px', borderRadius: '8px' }}
                  />
                </div>
              )}

              {isTranscribingFile && (
                <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--accent-cyan)', fontSize: '0.9rem', fontWeight: 600 }}>
                  <div className="pulse-icon" style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--accent-cyan)' }} />
                  Trascrizione con SenseVoice Small in corso...
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
                {activeMode === 'realtime' ? 'Trascrizione Live in Corso' : 'Testo Trascritto'}
              </span>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className="btn-secondary"
                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
                  onClick={() => handleCopy(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                  disabled={!(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                >
                  {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                  {copied ? 'Copiato!' : 'Copia'}
                </button>
                <button
                  className="btn-secondary"
                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
                  onClick={() => handleExport(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                  disabled={!(activeMode === 'realtime' ? liveTranscript : fileTranscript?.text)}
                >
                  <Download size={14} /> Esporta
                </button>
                <button
                  className="btn-secondary"
                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}
                  onClick={() => { setLiveTranscript(''); setFileTranscript(null); }}
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
                ? (liveTranscript || (isStreaming ? 'In attesa di audio...' : 'Premi il microfono e inizia a parlare.'))
                : (fileTranscript?.text || (isTranscribingFile ? 'Trascrizione in corso...' : 'Carica un file per visualizzare il testo trascritto.'))}
            </div>

            {/* File metadata card */}
            {activeMode === 'file' && fileTranscript && (
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', paddingTop: '0.25rem' }}>
                <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>
                  {getLanguageLabel(fileTranscript.language)}
                </span>
                <span className="badge badge-info" style={{ fontSize: '0.75rem' }}>
                  {getEmotionBadge(fileTranscript.emotion).label}
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
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.92rem', color: 'var(--text-main)' }}>
              <Clock size={16} color="var(--accent-purple)" />
              <span>Storico Trascrizioni</span>
            </div>
            {streamHistory.length > 0 && (
              <button
                onClick={() => setStreamHistory([])}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.78rem' }}
              >
                Svuota
              </button>
            )}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {streamHistory.length === 0 ? (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', padding: '2rem 1rem' }}>
                Nessuna frase registrata in questa sessione.
              </div>
            ) : (
              streamHistory.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '10px',
                    padding: '0.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    <span>{item.time}</span>
                    <span style={{ color: getEmotionBadge(item.emotion).color }}>
                      {getEmotionBadge(item.emotion).label}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.88rem', color: 'var(--text-main)', lineHeight: 1.4 }}>
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
