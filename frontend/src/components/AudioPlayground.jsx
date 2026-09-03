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
  VolumeX,
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
  Server,
  UserCheck,
  RotateCw,
  Square
} from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';

export class GaplessPCMPlayer {
  constructor(sampleRate = 44100, prebufferChunks = 2) {
    this.sampleRate = sampleRate;
    this.prebufferChunks = Math.max(1, Math.min(5, Number(prebufferChunks) || 2));
    this.audioCtx = null;
    this.nextStartTime = 0;
    this.isPlaying = false;
    this.activeNodes = [];
    this.wordTimers = [];
    this.pendingChunks = [];
    this.onChunkStarted = null;
    this.onWordStarted = null;
    this.onChunkPlayed = null;
    this.onChunkScheduled = null;
    this.onEnded = null;
  }

  setPrebufferChunks(n) {
    this.prebufferChunks = Math.max(1, Math.min(5, Number(n) || 1));
  }

  init() {
    if (!this.audioCtx || this.audioCtx.state === 'closed') {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContextClass({ sampleRate: this.sampleRate });
      this.nextStartTime = this.audioCtx.currentTime;
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  _scheduleBuffer(arrayBuffer, meta = {}) {
    this.init();
    if (!this.audioCtx) return 0;

    let int16Array;
    if (arrayBuffer instanceof Int16Array) {
      int16Array = arrayBuffer;
    } else if (arrayBuffer instanceof ArrayBuffer) {
      int16Array = new Int16Array(arrayBuffer);
    } else if (ArrayBuffer.isView(arrayBuffer)) {
      int16Array = new Int16Array(arrayBuffer.buffer, arrayBuffer.byteOffset, arrayBuffer.byteLength / 2);
    } else {
      return 0;
    }

    if (int16Array.length === 0) return 0;

    const float32Data = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Data[i] = int16Array[i] / 32768.0;
    }

    const audioBuffer = this.audioCtx.createBuffer(1, float32Data.length, this.sampleRate);
    audioBuffer.copyToChannel(float32Data, 0);

    const sourceNode = this.audioCtx.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(this.audioCtx.destination);

    const currentTime = this.audioCtx.currentTime;
    const startTime = Math.max(currentTime + 0.015, this.nextStartTime);
    sourceNode.start(startTime);

    // Schedule live karaoke subtitle trigger precisely when playback begins
    const delayMs = Math.max(0, (startTime - currentTime) * 1000.0);
    const chunkTimer = setTimeout(() => {
      if (this.onChunkStarted) this.onChunkStarted(meta);
    }, delayMs);
    this.wordTimers.push(chunkTimer);

    // Schedule word-by-word timestamps
    if (Array.isArray(meta.word_timestamps)) {
      meta.word_timestamps.forEach((wt, wIdx) => {
        const wordDelayMs = Math.max(0, (startTime + (wt.start_ms / 1000.0) - currentTime) * 1000.0);
        const wTimer = setTimeout(() => {
          if (this.onWordStarted) this.onWordStarted(wt, meta, wIdx);
        }, wordDelayMs);
        this.wordTimers.push(wTimer);
      });
    }

    this.nextStartTime = startTime + audioBuffer.duration;
    this.isPlaying = true;
    this.activeNodes.push(sourceNode);

    if (this.onChunkScheduled) {
      this.onChunkScheduled(meta, startTime, audioBuffer.duration, arrayBuffer);
    }

    sourceNode.onended = () => {
      const idx = this.activeNodes.indexOf(sourceNode);
      if (idx !== -1) this.activeNodes.splice(idx, 1);
      if (this.onChunkPlayed) this.onChunkPlayed(meta);
      if (this.activeNodes.length === 0 && (!this.audioCtx || this.audioCtx.currentTime >= this.nextStartTime - 0.05)) {
        this.isPlaying = false;
        if (this.onEnded) this.onEnded();
      }
    };

    return startTime;
  }

  flushPendingPrebuffer() {
    if (this.pendingChunks.length === 0) return;
    const toSchedule = [...this.pendingChunks];
    this.pendingChunks = [];
    for (const item of toSchedule) {
      this._scheduleBuffer(item.arrayBuffer, item.meta);
    }
  }

  enqueuePCM16(arrayBuffer, meta = {}) {
    if (this.isPlaying || this.prebufferChunks <= 1) {
      if (this.pendingChunks.length > 0) {
        this.flushPendingPrebuffer();
      }
      return this._scheduleBuffer(arrayBuffer, meta);
    }

    this.pendingChunks.push({ arrayBuffer, meta });

    if (this.pendingChunks.length >= this.prebufferChunks) {
      this.flushPendingPrebuffer();
      return this.nextStartTime;
    }

    return 0;
  }

  getCurrentTime() {
    return this.audioCtx ? this.audioCtx.currentTime : 0;
  }

  stopAndFlush() {
    this.pendingChunks = [];
    for (const t of this.wordTimers) {
      clearTimeout(t);
    }
    this.wordTimers = [];

    for (const node of this.activeNodes) {
      try {
        node.stop();
        node.disconnect();
      } catch (e) {}
    }
    this.activeNodes = [];
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      this.nextStartTime = this.audioCtx.currentTime;
    }
    this.isPlaying = false;
  }

  close() {
    this.stopAndFlush();
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      try {
        this.audioCtx.close();
      } catch (e) {}
      this.audioCtx = null;
    }
  }
}

export default function AudioPlayground({ apiBase, status, activeModel, isServerRunning, models = [], onNavigateToControl }) {
  const { t } = useTranslation();

  const isTtsModel = Boolean(
    activeModel?.includes('audio8') ||
    models.find(m => m.id === activeModel)?.task === 'text-to-speech' ||
    status?.model?.includes('audio8')
  );

  // Mode: 'realtime' (Live mic stream) | 'file' (Audio upload) | 'tts' (Text to Speech)
  const [activeMode, setActiveMode] = useState(() => isTtsModel ? 'tts' : 'realtime');

  useEffect(() => {
    if (isTtsModel) {
      setActiveMode('tts');
      if (activeModel) setTtsModel(activeModel);
    }
  }, [activeModel, isTtsModel]);
  
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

  // TTS State
  const [ttsText, setTtsText] = useState('Ciao! Questo è un test del sintetizzatore vocale su Alveare con AMD Ryzen AI.');
  const [ttsModel, setTtsModel] = useState('audio8-0.1b');
  const [ttsDevice, setTtsDevice] = useState(() => status?.device || 'npu');
  const [ttsRefAudioFile, setTtsRefAudioFile] = useState(null);
  const [ttsRefText, setTtsRefText] = useState('');
  const [ttsVoiceCloningOpen, setTtsVoiceCloningOpen] = useState(false);
  const [ttsMaxTokens, setTtsMaxTokens] = useState(300);
  const [ttsTemperature, setTtsTemperature] = useState(0.8);
  const [ttsTopP, setTtsTopP] = useState(0.95);
  const [isGeneratingTts, setIsGeneratingTts] = useState(false);
  const [ttsResult, setTtsResult] = useState(null);

  // TTS Streaming Real-Time State
  const [ttsStreamingEnabled, setTtsStreamingEnabled] = useState(false);
  const [ttsChunkStrategy, setTtsChunkStrategy] = useState('sentences');
  const [ttsWordsPerChunk, setTtsWordsPerChunk] = useState(14);
  const [ttsPrebufferChunks, setTtsPrebufferChunks] = useState(2);
  const [ttsTokenDelay, setTtsTokenDelay] = useState(0);
  const [ttsIsStreaming, setTtsIsStreaming] = useState(false);
  const [ttsStreamingStatus, setTtsStreamingStatus] = useState('idle');
  const [ttsActiveSpeakingText, setTtsActiveSpeakingText] = useState('');
  const [ttsPlayedSegments, setTtsPlayedSegments] = useState([]);
  const ttsPendingMetaRef = useRef(null);
  const ttsServerCompletedRef = useRef(false);
  const [ttsStreamMetrics, setTtsStreamMetrics] = useState({
    ttfaMs: null,
    chunksReceived: 0,
    chunksPlayed: 0,
    totalDurationSec: null,
    rtf: null,
    currentSegment: ''
  });

  // Word-by-word karaoke synchronization state driven by hardware AudioContext clock
  const [karaokeTokens, setKaraokeTokens] = useState([]);
  const [activeWordIndex, setActiveWordIndex] = useState(-1);
  const [currentAudioTime, setCurrentAudioTime] = useState(0);
  const wordTimelineRef = useRef([]);
  const nextWordIndexRef = useRef(0);
  const karaokeAnimRef = useRef(null);
  const currentAudioTimeRef = useRef(0);
  const activeWordRef = useRef(null);

  const tokenizeTextForKaraoke = useCallback((text) => {
    if (!text) return [];
    const parts = text.split(/(\s+)/);
    return parts.map((part, index) => ({
      id: index,
      text: part,
      isWord: part.trim().length > 0,
      startTime: null,
      endTime: null
    }));
  }, []);

  useEffect(() => {
    if (!ttsIsStreaming && ttsStreamingStatus !== 'playing' && ttsStreamingStatus !== 'completed') {
      const tokens = tokenizeTextForKaraoke(ttsText);
      setKaraokeTokens(tokens);
      wordTimelineRef.current = tokens;
      nextWordIndexRef.current = 0;
      setActiveWordIndex(-1);
    }
  }, [ttsText, tokenizeTextForKaraoke, ttsIsStreaming, ttsStreamingStatus]);

  useEffect(() => {
    if (activeWordRef.current && activeWordIndex !== -1) {
      try {
        activeWordRef.current.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
          inline: 'nearest'
        });
      } catch {}
    }
  }, [activeWordIndex]);

  useEffect(() => {
    if (!ttsIsStreaming && ttsStreamingStatus !== 'playing') {
      if (karaokeAnimRef.current) {
        cancelAnimationFrame(karaokeAnimRef.current);
        karaokeAnimRef.current = null;
      }
      return;
    }

    const tick = () => {
      if (gaplessPlayerRef.current && gaplessPlayerRef.current.audioCtx) {
        const nowCtx = gaplessPlayerRef.current.audioCtx.currentTime;
        currentAudioTimeRef.current = nowCtx;
        setCurrentAudioTime(nowCtx);

        const timeline = wordTimelineRef.current;
        let foundIdx = -1;
        for (let i = 0; i < timeline.length; i++) {
          const item = timeline[i];
          if (item.isWord && item.startTime !== null && item.endTime !== null) {
            if (nowCtx >= item.startTime && nowCtx <= item.endTime) {
              foundIdx = i;
              break;
            }
          }
        }
        setActiveWordIndex(foundIdx);
      }
      karaokeAnimRef.current = requestAnimationFrame(tick);
    };

    karaokeAnimRef.current = requestAnimationFrame(tick);
    return () => {
      if (karaokeAnimRef.current) {
        cancelAnimationFrame(karaokeAnimRef.current);
        karaokeAnimRef.current = null;
      }
    };
  }, [ttsIsStreaming, ttsStreamingStatus]);

  const [ttsHistory, setTtsHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('alveare_tts_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('alveare_tts_history', JSON.stringify(ttsHistory));
    } catch {}
  }, [ttsHistory]);

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
  const ttsAudioElementRef = useRef(null);
  const fileInputRef = useRef(null);
  const ttsFileInputRef = useRef(null);

  // TTS Streaming Refs
  const ttsWsRef = useRef(null);
  const gaplessPlayerRef = useRef(null);
  const ttsSimTimerRef = useRef(null);
  const ttsStartTimeRef = useRef(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopRealtimeStream();
      stopTtsStreaming();
      if (karaokeAnimRef.current) {
        cancelAnimationFrame(karaokeAnimRef.current);
        karaokeAnimRef.current = null;
      }
      if (gaplessPlayerRef.current) {
        gaplessPlayerRef.current.close();
      }
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

        // Gradient color for bars (cyan to purple)
        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
        gradient.addColorStop(0, '#06b6d4');
        gradient.addColorStop(1, '#8b5cf6');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1.5;
      }
    };

    renderFrame();
  }, []);

  // Start Real-Time WebSocket Streaming
  const startRealtimeStream = async () => {
    if (!isServerRunning) {
      setErrorMsg(t('audioPlayground.serverOffBannerDesc'));
      return;
    }
    setErrorMsg('');
    setIsFinalizing(false);

    try {
      // 1. Request microphone access (16kHz preferred)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      mediaStreamRef.current = stream;

      // 2. Setup AudioContext and Analyser for visualizer
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);

      // 3. Connect WebSocket to /ws/stt
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = apiBase ? apiBase.replace(/^http(s)?:\/\//, '') : window.location.host;
      const wsUrl = `${wsProtocol}//${wsHost}/ws/stt`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsStreaming(true);
        isStreamingRef.current = true;
        drawWaveform(analyser);

        // Send start handshake
        ws.send(JSON.stringify({
          action: "start",
          sample_rate: 16000,
          language: selectedLanguage,
          model: activeModel || "whisper-base"
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "ready") {
            // Stream ready
          } else if (data.type === "partial") {
            setLiveTranscript(data.text);
            setStreamStats({
              language: data.language || selectedLanguage,
              latency_ms: data.latency_ms || 0
            });
          } else if (data.type === "final") {
            if (data.text && data.text.trim()) {
              setStreamHistory(prev => [
                {
                  id: Date.now() + Math.random(),
                  text: data.text.trim(),
                  time: new Date().toLocaleTimeString(),
                  language: data.language || selectedLanguage,
                  latency_ms: data.latency_ms || 0
                },
                ...prev
              ]);
              setLiveTranscript('');
            }
            if (isStreamingRef.current === false) {
              setIsFinalizing(false);
            }
          } else if (data.type === "error") {
            setErrorMsg(`STT Stream Error: ${data.message}`);
            setIsFinalizing(false);
          }
        } catch (e) {
          console.error("Error parsing STT message:", e);
        }
      };

      ws.onerror = (e) => {
        console.error("STT WebSocket Error:", e);
        setErrorMsg("Errore di connessione WebSocket al server STT.");
        stopRealtimeStream();
      };

      ws.onclose = () => {
        setIsStreaming(false);
        isStreamingRef.current = false;
        setIsFinalizing(false);
      };

      // 4. Setup ScriptProcessorNode to capture PCM 16-bit 16kHz audio chunks
      const bufferSize = 2048;
      const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!isStreamingRef.current || ws.readyState !== WebSocket.OPEN) return;
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Convert float32 array to int16 PCM
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Send binary PCM frame over WebSocket
        ws.send(pcm16.buffer);
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

    } catch (err) {
      console.error("Failed to start audio stream:", err);
      setErrorMsg(t('audioPlayground.micError', { error: err.message }));
      stopRealtimeStream();
    }
  };

  // Stop Real-Time WebSocket Streaming
  const stopRealtimeStream = () => {
    isStreamingRef.current = false;
    setIsStreaming(false);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setIsFinalizing(true);
      try {
        wsRef.current.send(JSON.stringify({ action: "stop" }));
      } catch (e) {}
      setTimeout(() => {
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
        setIsFinalizing(false);
      }, 1200);
    } else {
      setIsFinalizing(false);
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext ? canvasRef.current.getContext('2d') : null;
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
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

  // Handle TTS Synthesis
  const handleTtsGenerate = async () => {
    if (!ttsText.trim()) return;
    if (!isServerRunning) {
      setErrorMsg(t('audioPlayground.serverOffBannerDesc'));
      return;
    }
    setIsGeneratingTts(true);
    setErrorMsg('');
    try {
      const formData = new FormData();
      formData.append('text', ttsText);
      formData.append('model', ttsModel);
      formData.append('device', ttsDevice);
      if (ttsVoiceCloningOpen && ttsRefAudioFile && ttsRefText.trim()) {
        formData.append('reference_audio', ttsRefAudioFile);
        formData.append('reference_text', ttsRefText.trim());
      }
      formData.append('max_new_tokens', ttsMaxTokens);
      formData.append('temperature', ttsTemperature);
      formData.append('top_p', ttsTopP);
      formData.append('prebuffer_chunks', ttsPrebufferChunks);

      const res = await fetch(`${apiBase}/api/tts/generate`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Sintesi vocale fallita');
      }
      const data = await res.json();
      setTtsResult(data);

      const newHistoryItem = {
        id: Date.now(),
        text: ttsText,
        time: new Date().toLocaleTimeString(),
        model: ttsModel,
        device: data.device || ttsDevice,
        duration_sec: data.duration_sec,
        latency_ms: data.latency_ms,
        rtf: data.rtf,
        tokens_per_sec: data.tokens_per_sec,
        num_tokens: data.num_tokens,
        audio_url: data.audio_url ? `${apiBase}${data.audio_url}` : ''
      };
      setTtsHistory(prev => [newHistoryItem, ...prev]);
    } catch (e) {
      console.error('TTS error:', e);
      setErrorMsg(e.message);
    } finally {
      setIsGeneratingTts(false);
    }
  };

  const startTtsStreaming = () => {
    if (!ttsText.trim()) return;
    if (!isServerRunning) {
      setErrorMsg(t('audioPlayground.serverOffBannerDesc'));
      return;
    }
    setErrorMsg('');
    setTtsIsStreaming(true);
    setTtsStreamingStatus('connecting');
    setTtsActiveSpeakingText('');
    setTtsPlayedSegments([]);
    
    // Initialize full text karaoke tokens with clean state
    const freshTokens = tokenizeTextForKaraoke(ttsText);
    setKaraokeTokens(freshTokens);
    wordTimelineRef.current = freshTokens;
    nextWordIndexRef.current = 0;
    setActiveWordIndex(-1);
    setCurrentAudioTime(0);
    currentAudioTimeRef.current = 0;

    setTtsStreamMetrics({
      ttfaMs: null,
      chunksReceived: 0,
      chunksPlayed: 0,
      totalDurationSec: null,
      rtf: null,
      currentSegment: ''
    });

    ttsServerCompletedRef.current = false;
    if (!gaplessPlayerRef.current) {
      gaplessPlayerRef.current = new GaplessPCMPlayer(44100, ttsPrebufferChunks);
    } else {
      gaplessPlayerRef.current.setPrebufferChunks(ttsPrebufferChunks);
      gaplessPlayerRef.current.init();
    }
    gaplessPlayerRef.current.onChunkStarted = (meta) => {
      if (meta && meta.text) {
        setTtsActiveSpeakingText(meta.text);
      }
    };
    gaplessPlayerRef.current.onChunkPlayed = (meta) => {
      setTtsStreamMetrics(prev => ({
        ...prev,
        chunksPlayed: prev.chunksPlayed + 1
      }));
      if (meta && meta.text) {
        setTtsPlayedSegments(prev => [...prev, meta.text]);
      }
    };
    gaplessPlayerRef.current.onChunkScheduled = (meta, chunkStartTime, durationSec, rawData) => {
      const wordTimestamps = meta.word_timestamps || [];
      const timeline = [...wordTimelineRef.current];

      if (wordTimestamps.length > 0) {
        let wtIdx = 0;
        for (let i = nextWordIndexRef.current; i < timeline.length && wtIdx < wordTimestamps.length; i++) {
          if (timeline[i].isWord) {
            const wt = wordTimestamps[wtIdx];
            timeline[i].startTime = chunkStartTime + (wt.start_ms / 1000.0);
            timeline[i].endTime = chunkStartTime + (wt.end_ms / 1000.0);
            wtIdx++;
            nextWordIndexRef.current = i + 1;
          }
        }
      } else if (meta.text) {
        const chunkWords = meta.text.trim().split(/\s+/);
        const durSec = durationSec || (meta.duration_ms ? meta.duration_ms / 1000.0 : (rawData ? (rawData.byteLength / 2) / 44100 : 1.0));
        const wordDur = durSec / Math.max(1, chunkWords.length);
        let cwIdx = 0;
        for (let i = nextWordIndexRef.current; i < timeline.length && cwIdx < chunkWords.length; i++) {
          if (timeline[i].isWord) {
            timeline[i].startTime = chunkStartTime + cwIdx * wordDur;
            timeline[i].endTime = chunkStartTime + (cwIdx + 1) * wordDur;
            cwIdx++;
            nextWordIndexRef.current = i + 1;
          }
        }
      }

      wordTimelineRef.current = timeline;
      setKaraokeTokens([...timeline]);
    };
    gaplessPlayerRef.current.onEnded = () => {
      if (ttsServerCompletedRef.current) {
        setTtsStreamingStatus('completed');
        setTtsIsStreaming(false);
        setTtsActiveSpeakingText('');
        setActiveWordIndex(-1);
      } else {
        setTtsStreamingStatus('buffering');
        setTtsActiveSpeakingText('');
      }
    };

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = apiBase ? apiBase.replace(/^http(s)?:\/\//, '') : window.location.host;
    const wsUrl = `${wsProtocol}//${wsHost}/ws/tts`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ttsWsRef.current = ws;

    ws.onopen = () => {
      ttsStartTimeRef.current = performance.now();
      setTtsStreamingStatus('streaming');

      // Send start handshake
      ws.send(JSON.stringify({
        action: "start",
        voice: "valeria",
        model: ttsModel,
        device: ttsDevice,
        sample_rate: 44100,
        format: "pcm16",
        speed: 1.0,
        pitch: 0.0,
        temperature: ttsTemperature,
        chunk_strategy: ttsChunkStrategy,
        words_per_chunk: ttsWordsPerChunk,
        prebuffer_chunks: ttsPrebufferChunks
      }));

      // Simulate LLM streaming token-by-token or send all
      if (ttsTokenDelay > 0) {
        const words = ttsText.split(/(\s+)/);
        let idx = 0;
        ttsSimTimerRef.current = setInterval(() => {
          if (idx < words.length && ws.readyState === WebSocket.OPEN) {
            const piece = words[idx];
            if (piece) {
              ws.send(JSON.stringify({
                action: "text_chunk",
                text: piece,
                is_final: false
              }));
            }
            idx++;
          } else {
            if (ttsSimTimerRef.current) {
              clearInterval(ttsSimTimerRef.current);
              ttsSimTimerRef.current = null;
            }
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ action: "flush" }));
            }
          }
        }, ttsTokenDelay);
      } else {
        ws.send(JSON.stringify({
          action: "text_chunk",
          text: ttsText,
          is_final: true
        }));
      }
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const now = performance.now();
        const ttfa = ttsStartTimeRef.current ? Math.round(now - ttsStartTimeRef.current) : 0;

        setTtsStreamMetrics(prev => ({
          ...prev,
          ttfaMs: prev.ttfaMs !== null ? prev.ttfaMs : ttfa,
          chunksReceived: prev.chunksReceived + 1
        }));

        const meta = ttsPendingMetaRef.current || { text: '', word_timestamps: [] };
        ttsPendingMetaRef.current = null;

        if (gaplessPlayerRef.current) {
          gaplessPlayerRef.current.enqueuePCM16(event.data, meta);
          if (gaplessPlayerRef.current.isPlaying) {
            setTtsStreamingStatus('playing');
          } else {
            setTtsStreamingStatus('buffering');
          }
        }
      } else if (typeof event.data === 'string') {
        try {
          const data = JSON.parse(event.data);
          if (data.event === "chunk_meta") {
            ttsPendingMetaRef.current = data;
            setTtsStreamMetrics(prev => ({
              ...prev,
              currentSegment: data.text || prev.currentSegment
            }));
          } else if (data.event === "completed") {
            ttsServerCompletedRef.current = true;
            if (gaplessPlayerRef.current) {
              gaplessPlayerRef.current.flushPendingPrebuffer();
            }
            setTtsStreamMetrics(prev => ({
              ...prev,
              totalDurationSec: data.total_duration_sec,
              rtf: data.rtf
            }));

            const isPlayerBusy = gaplessPlayerRef.current && (
              gaplessPlayerRef.current.isPlaying ||
              (gaplessPlayerRef.current.activeNodes && gaplessPlayerRef.current.activeNodes.length > 0) ||
              (gaplessPlayerRef.current.pendingChunks && gaplessPlayerRef.current.pendingChunks.length > 0)
            );

            if (!isPlayerBusy) {
              setTtsStreamingStatus('completed');
              setTtsIsStreaming(false);
              setTtsActiveSpeakingText('');
              setActiveWordIndex(-1);
            }

            // Record to TTS History
            const newHistoryItem = {
              id: Date.now(),
              text: ttsText,
              time: new Date().toLocaleTimeString(),
              model: ttsModel,
              device: ttsDevice,
              duration_sec: data.total_duration_sec,
              latency_ms: data.latency_ttfa_ms || Math.round(performance.now() - (ttsStartTimeRef.current || performance.now())),
              rtf: data.rtf,
              tokens_per_sec: data.tokens_per_sec || 0,
              num_tokens: data.num_tokens || 0,
              is_stream: true
            };
            setTtsHistory(prev => [newHistoryItem, ...prev]);
          } else if (data.event === "interrupted") {
            setTtsStreamingStatus('interrupted');
            setTtsIsStreaming(false);
            setActiveWordIndex(-1);
          } else if (data.event === "error") {
            setErrorMsg(data.message || "Errore nello streaming TTS");
            setTtsIsStreaming(false);
            setActiveWordIndex(-1);
          }
        } catch (e) {
          console.error("TTS WS message parse error:", e);
        }
      }
    };

    ws.onerror = (e) => {
      console.error("TTS WebSocket error:", e);
      setErrorMsg("Errore di connessione WebSocket al server TTS.");
      stopTtsStreaming();
    };

    ws.onclose = () => {
      // Keep playing queued audio until finished
    };
  };

  const stopTtsStreaming = () => {
    if (ttsSimTimerRef.current) {
      clearInterval(ttsSimTimerRef.current);
      ttsSimTimerRef.current = null;
    }
    if (ttsWsRef.current && ttsWsRef.current.readyState === WebSocket.OPEN) {
      try {
        ttsWsRef.current.send(JSON.stringify({ action: "interrupt", reason: "user_barge_in" }));
      } catch (e) {}
      setTimeout(() => {
        if (ttsWsRef.current) {
          ttsWsRef.current.close();
          ttsWsRef.current = null;
        }
      }, 200);
    }
    if (gaplessPlayerRef.current) {
      gaplessPlayerRef.current.stopAndFlush();
    }
    ttsServerCompletedRef.current = true;
    setTtsStreamingStatus('interrupted');
    setTtsIsStreaming(false);
    setTtsActiveSpeakingText('');
    setActiveWordIndex(-1);
  };

  const getLanguageLabel = (lang) => {
    switch (lang) {
      case 'it': return '🇮🇹 Italiano (IT)';
      case 'en': return '🇬🇧 English (EN)';
      case 'zh': return '🇨🇳 中文 (ZH)';
      case 'de': return '🇩🇪 Deutsch (DE)';
      case 'es': return '🇪🇸 Español (ES)';
      case 'fr': return '🇫🇷 Français (FR)';
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
                {activeMode === 'tts' ? (t('audioPlayground.ttsTitle') || 'Audio8 Neural TTS Studio') : t('audioPlayground.title')}
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
            <p style={{ margin: '0.15rem 0 0', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              {activeMode === 'tts' ? (t('audioPlayground.ttsSubtitle') || 'Sintesi vocale neurale e voice cloning zero-shot su AMD Ryzen AI NPU e CPU.') : t('audioPlayground.subtitle')}
            </p>
          </div>
        </div>

        {/* Global Toolbar Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          
          {/* Spoken Language Selector (Only for STT) */}
          {activeMode !== 'tts' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-secondary)', padding: '0.35rem 0.75rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
              <Globe size={15} color="var(--accent-cyan)" />
              <label htmlFor="stt-lang-select" style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                {t('audioPlayground.languageSelectLabel')}
              </label>
              <select
                id="stt-lang-select"
                value={selectedLanguage}
                onChange={(e) => handleLanguageChange(e.target.value)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-main)',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  outline: 'none'
                }}
              >
                <option value="auto" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>{t('audioPlayground.langAuto')}</option>
                <option value="it" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>{t('audioPlayground.langIt')}</option>
                <option value="en" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>{t('audioPlayground.langEn')}</option>
                <option value="zh" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>{t('audioPlayground.langZh')}</option>
                <option value="de" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇩🇪 Deutsch</option>
                <option value="es" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇪🇸 Español</option>
                <option value="fr" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇫🇷 Français</option>
                <option value="ja" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇯🇵 日本語</option>
                <option value="ko" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇰🇷 한국어</option>
                <option value="ru" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇷🇺 Русский</option>
                <option value="ar" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇸🇦 العربية</option>
                <option value="nl" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇳🇱 Nederlands</option>
                <option value="pl" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇵🇱 Polski</option>
                <option value="tr" style={{ background: 'var(--bg-secondary)', color: 'var(--text-main)' }}>🇹🇷 Türkçe</option>
              </select>
            </div>
          )}

          {/* Mode Switcher Tabs */}
          <div style={{ display: 'flex', background: 'var(--bg-secondary)', padding: '0.3rem', borderRadius: '10px', border: '1px solid var(--border-color)', gap: '0.2rem' }}>
            {!isTtsModel && (
              <>
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
              </>
            )}
            {isTtsModel && (
              <button
                onClick={() => { setActiveMode('tts'); stopRealtimeStream(); }}
                style={{
                  padding: '0.45rem 0.85rem',
                  borderRadius: '7px',
                  border: 'none',
                  background: activeMode === 'tts' ? 'var(--gradient-brand)' : 'transparent',
                  color: activeMode === 'tts' ? '#fff' : 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  transition: 'all 0.2s ease'
                }}
              >
                <Volume2 size={15} /> {t('audioPlayground.ttsTab') || 'Sintesi Vocale (TTS)'}
              </button>
            )}
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
          ) : activeMode === 'file' ? (
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
          ) : (
            /* TEXT TO SPEECH (TTS) STUDIO */
            <div style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '16px',
              padding: '1.5rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.25rem',
              boxShadow: '0 8px 30px rgba(0,0,0,0.15)'
            }}>
              
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Volume2 size={18} color="var(--accent-cyan)" />
                  <label style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    {t('audioPlayground.ttsInputPrompt') || 'Testo da Sintetizzare'}
                  </label>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  {/* Informative Device Badge */}
                  <div
                    style={{
                      background: 'rgba(0,0,0,0.3)',
                      border: '1px solid var(--border-color)',
                      color: (status?.device === 'npu' || ttsDevice === 'npu') ? '#34d399' : '#60a5fa',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      padding: '0.35rem 0.75rem',
                      borderRadius: '8px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.35rem'
                    }}
                    title="Dispositivo hardware configurato dal Pannello di Controllo"
                  >
                    {(status?.device === 'npu' || ttsDevice === 'npu') ? (
                      <>
                        <Zap size={13} color="#34d399" /> ⚡ NPU (XDNA2)
                      </>
                    ) : (
                      <>
                        <Server size={13} color="#60a5fa" /> 🖥️ CPU
                      </>
                    )}
                  </div>

                  <div
                    style={{
                      background: 'rgba(0,0,0,0.3)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-main)',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      padding: '0.35rem 0.75rem',
                      borderRadius: '8px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem'
                    }}
                    title="Il modello può essere configurato e modificato dal Pannello di Controllo"
                  >
                    <span style={{ color: 'var(--accent-cyan)' }}>●</span> {status?.model || activeModel || 'audio8-0.1b'}
                  </div>

                  <button
                    type="button"
                    onClick={() => setTtsStreamingEnabled(!ttsStreamingEnabled)}
                    style={{
                      padding: '0.35rem 0.75rem',
                      borderRadius: '8px',
                      border: ttsStreamingEnabled ? '1px solid rgba(6, 182, 212, 0.6)' : '1px solid var(--border-color)',
                      background: ttsStreamingEnabled ? 'linear-gradient(135deg, rgba(6, 182, 212, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%)' : 'transparent',
                      color: ttsStreamingEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)',
                      fontSize: '0.78rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      transition: 'all 0.2s ease'
                    }}
                    title="Abilita la generazione e riproduzione in streaming WebSocket in tempo reale"
                  >
                    <Zap size={14} color={ttsStreamingEnabled ? 'var(--accent-cyan)' : 'currentColor'} />
                    {ttsStreamingEnabled ? '⚡ Streaming Attivo' : '📡 Modalità Streaming'}
                  </button>

                  <button
                    type="button"
                    onClick={() => setTtsVoiceCloningOpen(!ttsVoiceCloningOpen)}
                    style={{
                      padding: '0.35rem 0.75rem',
                      borderRadius: '8px',
                      border: '1px solid var(--border-color)',
                      background: ttsVoiceCloningOpen ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                      color: ttsVoiceCloningOpen ? 'var(--accent-cyan)' : 'var(--text-muted)',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem'
                    }}
                  >
                    <UserCheck size={14} /> {t('audioPlayground.ttsVoiceCloningTitle') || 'Voice Cloning'}
                  </button>
                </div>
              </div>

              {/* Text Input Area */}
              <textarea
                value={ttsText}
                onChange={(e) => setTtsText(e.target.value)}
                placeholder={t('audioPlayground.ttsInputPlaceholder') || 'Inserisci qui il testo in italiano o in un\'altra lingua...'}
                rows={4}
                style={{
                  width: '100%',
                  background: 'rgba(0,0,0,0.25)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '12px',
                  padding: '0.85rem',
                  fontSize: '0.95rem',
                  color: 'var(--text-main)',
                  resize: 'vertical',
                  outline: 'none',
                  lineHeight: 1.5
                }}
              />

              {/* Gemini Live Subtitle / Karaoke Teleprompter Display */}
              {(ttsStreamingEnabled || ttsIsStreaming || ttsStreamingStatus !== 'idle') && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.85) 100%)',
                  border: activeWordIndex !== -1 ? '1px solid rgba(6, 182, 212, 0.6)' : '1px solid var(--border-color)',
                  borderRadius: '14px',
                  padding: '1.1rem 1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                  boxShadow: activeWordIndex !== -1 ? '0 0 30px rgba(6, 182, 212, 0.25)' : 'none',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className={activeWordIndex !== -1 || ttsIsStreaming ? 'pulse-icon' : ''} style={{
                        width: '10px',
                        height: '10px',
                        borderRadius: '50%',
                        background: activeWordIndex !== -1 ? '#38bdf8' : ttsIsStreaming ? '#f59e0b' : 'var(--text-muted)'
                      }} />
                      <span style={{ fontSize: '0.82rem', fontWeight: 800, letterSpacing: '0.05em', color: activeWordIndex !== -1 ? 'var(--accent-cyan)' : 'var(--text-muted)', textTransform: 'uppercase' }}>
                        {activeWordIndex !== -1 ? '● LIVE AUDIO TELEPROMPTER' : ttsIsStreaming ? '● STREAMING LIVE' : 'SOTTOTITOLI LIVE AUDIO'}
                      </span>
                    </div>

                    {activeWordIndex !== -1 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <span style={{ width: '3px', height: '14px', background: '#38bdf8', borderRadius: '2px' }} />
                        <span style={{ width: '3px', height: '22px', background: '#818cf8', borderRadius: '2px' }} />
                        <span style={{ width: '3px', height: '10px', background: '#c084fc', borderRadius: '2px' }} />
                        <span style={{ width: '3px', height: '18px', background: '#38bdf8', borderRadius: '2px' }} />
                      </div>
                    )}
                  </div>

                  {/* Full Text Karaoke Highlighted Prompt Body */}
                  <div style={{
                    fontSize: '1.08rem',
                    lineHeight: 1.75,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    padding: '0.5rem 0',
                    minHeight: '4rem',
                    wordBreak: 'break-word',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {karaokeTokens.length > 0 ? (
                      <div>
                        {karaokeTokens.map((token) => {
                          if (!token.isWord) {
                            return <span key={token.id}>{token.text}</span>;
                          }

                          const isActive = token.id === activeWordIndex;
                          const isPlayed = token.endTime !== null && currentAudioTime > token.endTime;

                          if (isActive) {
                            return (
                              <span
                                key={token.id}
                                ref={activeWordRef}
                                style={{
                                  background: 'linear-gradient(90deg, rgba(6, 182, 212, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%)',
                                  color: '#38bdf8',
                                  fontWeight: 800,
                                  padding: '0.15rem 0.45rem',
                                  borderRadius: '6px',
                                  border: '1px solid rgba(56, 189, 248, 0.6)',
                                  boxShadow: '0 0 16px rgba(56, 189, 248, 0.45)',
                                  display: 'inline-block',
                                  transform: 'scale(1.05)',
                                  transition: 'all 0.1s ease',
                                  margin: '0 1px'
                                }}
                              >
                                {token.text}
                              </span>
                            );
                          } else if (isPlayed) {
                            return (
                              <span
                                key={token.id}
                                style={{
                                  color: 'rgba(148, 163, 184, 0.7)',
                                  transition: 'color 0.2s ease'
                                }}
                              >
                                {token.text}
                              </span>
                            );
                          } else {
                            return (
                              <span
                                key={token.id}
                                style={{
                                  color: 'var(--text-main, #f1f5f9)',
                                  transition: 'color 0.2s ease'
                                }}
                              >
                                {token.text}
                              </span>
                            );
                          }
                        })}
                        {ttsIsStreaming && activeWordIndex === -1 && ttsStreamingStatus === 'streaming' && (
                          <span style={{
                            color: '#f59e0b',
                            fontSize: '0.88rem',
                            fontStyle: 'italic',
                            marginLeft: '0.5rem',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.35rem'
                          }}>
                            ⏳ In elaborazione audio...
                          </span>
                        )}
                      </div>
                    ) : (
                      <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem' }}>
                        {ttsIsStreaming ? 'In attesa del primo blocco vocale dal motore neurale...' : 'Avvia la sintesi in streaming per visualizzare i sottotitoli sincronizzati in tempo reale.'}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Zero-Shot Voice Cloning Collapsible Card */}
              {ttsVoiceCloningOpen && (
                <div style={{
                  background: 'rgba(6, 182, 212, 0.05)',
                  border: '1px solid rgba(6, 182, 212, 0.25)',
                  borderRadius: '12px',
                  padding: '1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.85rem'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Sparkles size={16} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                      {t('audioPlayground.ttsVoiceCloningTitle') || 'Voice Cloning Zero-Shot'}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      ({t('audioPlayground.ttsVoiceCloningDesc') || 'Clona qualsiasi timbro vocale'})
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    <input
                      ref={ttsFileInputRef}
                      type="file"
                      accept="audio/*,.wav,.mp3"
                      style={{ display: 'none' }}
                      onChange={(e) => {
                        if (e.target.files?.[0]) {
                          setTtsRefAudioFile(e.target.files[0]);
                        }
                        e.target.value = '';
                      }}
                    />
                    <button
                      type="button"
                      className="btn-secondary"
                      style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
                      onClick={() => ttsFileInputRef.current?.click()}
                    >
                      <Upload size={14} /> {ttsRefAudioFile ? ttsRefAudioFile.name : (t('audioPlayground.ttsRefAudioLabel') || 'Carica Clip Vocale')}
                    </button>
                    {ttsRefAudioFile && (
                      <button
                        type="button"
                        onClick={() => setTtsRefAudioFile(null)}
                        style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer', fontSize: '0.78rem' }}
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>

                  <input
                    type="text"
                    value={ttsRefText}
                    onChange={(e) => setTtsRefText(e.target.value)}
                    placeholder={t('audioPlayground.ttsRefTextPlaceholder') || 'Trascrizione esatta delle parole pronunciate nella clip vocale...'}
                    style={{
                      width: '100%',
                      background: 'rgba(0,0,0,0.3)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      padding: '0.6rem 0.85rem',
                      fontSize: '0.85rem',
                      color: 'var(--text-main)',
                      outline: 'none'
                    }}
                  />
                </div>
              )}

              {/* Real-Time Streaming Configuration Card */}
              {ttsStreamingEnabled && (
                <div style={{
                  background: 'rgba(6, 182, 212, 0.05)',
                  border: '1px solid rgba(6, 182, 212, 0.25)',
                  borderRadius: '12px',
                  padding: '1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.85rem'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Radio size={15} /> Parametri Streaming Real-Time (WebSocket /ws/tts)
                    </span>
                    <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                      Riproduzione fluida gapless con Web Audio API
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '0.76rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                        Strategia di Chunking
                      </label>
                      <select
                        value={ttsChunkStrategy}
                        onChange={(e) => setTtsChunkStrategy(e.target.value)}
                        style={{
                          width: '100%',
                          background: 'rgba(0,0,0,0.3)',
                          border: '1px solid var(--border-color)',
                          borderRadius: '7px',
                          padding: '0.4rem 0.6rem',
                          color: 'var(--text-main)',
                          fontSize: '0.8rem',
                          fontWeight: 600
                        }}
                      >
                        <option value="sentences">Frasi Complete (Prosodia Naturale Audio8: . ! ? ; :)</option>
                        <option value="words">Buffer a {ttsWordsPerChunk} Parole</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: '0.76rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                        Pre-buffering di Avvio ({ttsPrebufferChunks} {ttsPrebufferChunks === 1 ? 'chunk' : 'chunk'})
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="5"
                        step="1"
                        aria-label="Pre-buffering di Avvio"
                        value={ttsPrebufferChunks}
                        onChange={(e) => setTtsPrebufferChunks(Number(e.target.value))}
                        style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: '0.76rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                        Simulazione Token LLM ({ttsTokenDelay === 0 ? 'Istantanea' : `${ttsTokenDelay} ms/token`})
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="80"
                        step="10"
                        value={ttsTokenDelay}
                        onChange={(e) => setTtsTokenDelay(Number(e.target.value))}
                        style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Generate Button & Progress */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
                  {ttsStreamingEnabled ? (
                    <>
                      <button
                        className="btn-primary"
                        onClick={startTtsStreaming}
                        disabled={ttsIsStreaming || !ttsText.trim() || !isServerRunning}
                        style={{
                          padding: '0.6rem 1.4rem',
                          fontSize: '0.9rem',
                          fontWeight: 700,
                          borderRadius: '10px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem',
                          boxShadow: '0 4px 15px rgba(6, 182, 212, 0.4)'
                        }}
                      >
                        {ttsIsStreaming ? <RotateCw size={16} className="spin-icon" /> : <Radio size={16} />}
                        {ttsIsStreaming ? 'Streaming in corso...' : 'Sintetizza in Streaming (Live Playback)'}
                      </button>

                      {ttsIsStreaming && (
                        <button
                          className="btn-secondary"
                          onClick={stopTtsStreaming}
                          style={{
                            padding: '0.6rem 1rem',
                            fontSize: '0.85rem',
                            fontWeight: 700,
                            borderRadius: '10px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem',
                            borderColor: 'rgba(239, 68, 68, 0.5)',
                            color: '#f87171'
                          }}
                        >
                          <VolumeX size={15} /> Interrompi (Barge-in)
                        </button>
                      )}
                    </>
                  ) : (
                    <button
                      className="btn-primary"
                      onClick={handleTtsGenerate}
                      disabled={isGeneratingTts || !ttsText.trim() || !isServerRunning}
                      style={{
                        padding: '0.6rem 1.4rem',
                        fontSize: '0.9rem',
                        fontWeight: 700,
                        borderRadius: '10px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        boxShadow: '0 4px 15px rgba(6, 182, 212, 0.4)'
                      }}
                    >
                      {isGeneratingTts ? <RotateCw size={16} className="spin-icon" /> : <Volume2 size={16} />}
                      {isGeneratingTts ? (t('audioPlayground.ttsGenerating') || 'Sintesi vocale in corso...') : (t('audioPlayground.ttsGenerateBtn') || 'Genera Audio')}
                    </button>
                  )}
                </div>

                {ttsStreamingEnabled && (ttsStreamMetrics.chunksReceived > 0 || ttsIsStreaming) && (
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    {ttsStreamMetrics.ttfaMs && (
                      <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>
                        ⚡ TTFA: {ttsStreamMetrics.ttfaMs} ms
                      </span>
                    )}
                    <span className="badge badge-cyan" style={{ fontSize: '0.75rem' }}>
                      📦 Chunk: {ttsStreamMetrics.chunksPlayed} / {ttsStreamMetrics.chunksReceived}
                    </span>
                    {ttsStreamMetrics.totalDurationSec && (
                      <span className="badge badge-purple" style={{ fontSize: '0.75rem' }}>
                        ⏳ Durata: {ttsStreamMetrics.totalDurationSec}s
                      </span>
                    )}
                  </div>
                )}

                {!ttsStreamingEnabled && ttsResult && (
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    <span
                      className="badge"
                      style={{
                        fontSize: '0.75rem',
                        background: (ttsResult.device || ttsDevice) === 'cpu' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                        color: (ttsResult.device || ttsDevice) === 'cpu' ? '#60a5fa' : '#34d399',
                        border: (ttsResult.device || ttsDevice) === 'cpu' ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid rgba(16, 185, 129, 0.4)'
                      }}
                    >
                      {(ttsResult.device || ttsDevice) === 'cpu' ? '🖥️ CPU' : '⚡ NPU (Offload)'}
                    </span>
                    <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>
                      ⏱️ {ttsResult.latency_ms} ms
                    </span>
                    <span className="badge badge-cyan" style={{ fontSize: '0.75rem' }}>
                      ⏳ {ttsResult.duration_sec}s
                    </span>
                    <span className="badge badge-purple" style={{ fontSize: '0.75rem' }}>
                      ⚡ RTF: {ttsResult.rtf}x
                    </span>
                    {ttsResult.tokens_per_sec > 0 && (
                      <span className="badge badge-info" style={{ fontSize: '0.75rem', background: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24', border: '1px solid rgba(245, 158, 11, 0.4)' }}>
                        🚀 {ttsResult.tokens_per_sec} tok/s
                      </span>
                    )}
                  </div>
                )}
              </div>

            </div>
          )}

          {/* Quick Realtime Live Transcript Box / TTS Result Player */}
          {activeMode === 'tts' ? (
            ttsResult && ttsResult.audio_url && (
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
                    <Volume2 size={16} color="var(--accent-cyan)" />
                    {t('audioPlayground.ttsResultTitle') || 'Audio Generato'}
                  </span>

                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <a
                      href={`${apiBase}${ttsResult.audio_url}`}
                      download="alveare_speech.wav"
                      className="btn-secondary"
                      style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
                    >
                      <Download size={14} /> {t('audioPlayground.ttsDownload') || 'Scarica WAV'}
                    </a>
                  </div>
                </div>

                <audio
                  ref={ttsAudioElementRef}
                  src={`${apiBase}${ttsResult.audio_url}`}
                  controls
                  autoPlay
                  style={{ width: '100%', height: '42px', borderRadius: '8px' }}
                />
              </div>
            )
          ) : (
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
          )}

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
              <span>{activeMode === 'tts' ? (t('audioPlayground.historyTitle') || 'Storico Sintesi Vocale') : t('audioPlayground.historyTitle')}</span>
              {((activeMode === 'tts' ? ttsHistory : streamHistory).length > 0) && (
                <span style={{ fontSize: '0.72rem', background: 'rgba(255,255,255,0.1)', padding: '0.1rem 0.45rem', borderRadius: '10px', color: 'var(--text-muted)' }}>
                  {(activeMode === 'tts' ? ttsHistory : streamHistory).length}
                </span>
              )}
            </div>
            {((activeMode === 'tts' ? ttsHistory : streamHistory).length > 0) && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <button
                  onClick={() => {
                    if (activeMode === 'tts') {
                      setTtsHistory([]);
                      try { localStorage.removeItem('alveare_tts_history'); } catch {}
                    } else {
                      setStreamHistory([]);
                      try { localStorage.removeItem('alveare_stt_history'); } catch {}
                    }
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
            {activeMode === 'tts' ? (
              ttsHistory.length === 0 ? (
                <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', padding: '2rem 1rem' }}>
                  Nessuna frase generata in questa sessione.
                </div>
              ) : (
                ttsHistory.map((item, idx) => (
                  <div
                    key={item.id || idx}
                    style={{
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '10px',
                      padding: '0.75rem',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.45rem'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', flexWrap: 'wrap', gap: '0.3rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <span>⏱️ {item.time}</span>
                        <span
                          style={{
                            fontSize: '0.68rem',
                            padding: '0.05rem 0.35rem',
                            borderRadius: '4px',
                            background: item.device === 'cpu' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                            color: item.device === 'cpu' ? '#60a5fa' : '#34d399'
                          }}
                        >
                          {item.device === 'cpu' ? 'CPU' : 'NPU'}
                        </span>
                      </div>
                      <span>⚡ {item.latency_ms} ms (RTF: {item.rtf}x{item.tokens_per_sec ? ` · ${item.tokens_per_sec} tok/s` : ''})</span>
                    </div>
                    <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--text-main)', lineHeight: 1.4 }}>
                      "{item.text}"
                    </p>
                    {item.audio_url && (
                      <audio src={item.audio_url} controls style={{ width: '100%', height: '32px', marginTop: '0.3rem' }} />
                    )}
                  </div>
                ))
              )
            ) : (
              streamHistory.length === 0 ? (
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
                      gap: '0.35rem'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      <span>⏱️ {item.time}</span>
                      <span>{getLanguageLabel(item.language)}</span>
                    </div>
                    <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--text-main)', lineHeight: 1.4 }}>
                      "{item.text}"
                    </p>
                    {item.latency_ms > 0 && (
                      <div style={{ fontSize: '0.7rem', color: 'var(--accent-cyan)', textAlign: 'right' }}>
                        {item.latency_ms} ms latenza
                      </div>
                    )}
                  </div>
                ))
              )
            )}
          </div>

        </div>

      </div>

    </div>
  );
}
