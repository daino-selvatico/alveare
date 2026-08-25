import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  User,
  Bot,
  Cpu,
  Zap,
  StopCircle,
  Copy,
  Check,
  Brain,
  ChevronDown,
  ChevronUp,
  Plus,
  Sparkles,
  Paperclip,
  Image as ImageIcon,
  Music,
  FileText,
  Upload,
  X,
  Sliders,
  AlertTriangle,
  RefreshCw,
  HardDrive,
  Mic,
  MicOff,
  Play,
  Volume2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SidebarHistory from './SidebarHistory';
import GenerationSettingsPanel from './GenerationSettingsPanel';
import {
  getConversations,
  getConversation,
  createConversation,
  saveConversation,
  deleteConversation,
  renameConversation,
  clearAllConversations,
  getActiveConversationId,
  setActiveConversationId,
  generateTitleFromMessage,
  getGlobalSettings,
  saveGlobalSettings,
  DEFAULT_SETTINGS
} from '../utils/chatStorage';
import { useTranslation } from '../i18n/I18nContext';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import { SlidingWindowTpsCalculator } from '../utils/tpsCalculator';
import { extractDocumentText } from '../utils/documentParser';


function parseThinking(content) {
  if (!content) return { thinking: '', text: '' };

  // 1. Gemma 4 channel tokens format: <|channel>thought...<channel|> or <channel>thought...
  const gemmaRegex = /(?:<\|channel\|?>thought\n?|<channel>thought\n?)([\s\S]*?)(?:<channel\|?>|<\|channel\|?>text\n?|<channel>text\n?|$)/i;
  const gemmaMatch = content.match(gemmaRegex);
  if (gemmaMatch) {
    const thinking = gemmaMatch[1].trim();
    let text = content.replace(gemmaRegex, '').trim();
    text = text.replace(/^(?:<\|channel\|?>text\n?|<channel>text\n?)/i, '').trim();
    return { thinking, text };
  }

  // 2. XML tags: <thought>...</thought> or <thinking>...</thinking>
  const xmlRegex = /<(?:thought|thinking)>([\s\S]*?)(?:<\/(?:thought|thinking)>|$)/i;
  const xmlMatch = content.match(xmlRegex);
  if (xmlMatch) {
    const thinking = xmlMatch[1].trim();
    const text = content.replace(/<(?:thought|thinking)>[\s\S]*?(?:<\/(?:thought|thinking)>|$)/gi, '').trim();
    return { thinking, text };
  }

  // 3. Simple inline thinking section: "Thinking Process:\n..."
  const headerRegex = /(?:Thinking Process|Pensiero):\n([\s\S]*?)(?:\n\n(?:Response|Risposta):\n|$)/i;
  const headerMatch = content.match(headerRegex);
  if (headerMatch) {
    const thinking = headerMatch[1].trim();
    const text = content.replace(headerRegex, '').trim();
    return { thinking, text };
  }

  // 4. Natural language chain-of-thought monologue: e.g. "I need to analyze... Plan:... <Answer>"
  const monologueRegex = /^((?:I need to analyze|Let's analyze|Constraint Check:|The user is asking)[\s\S]*?(?:Plan:\s*(?:[^\n]+\n)*\s*(?:\([^)]+\)\s*)?))([A-ZÀ-Úa-zà-ú].*)$/s;
  const monologueMatch = content.match(monologueRegex);
  if (monologueMatch) {
    const thinking = monologueMatch[1].trim();
    const text = monologueMatch[2].trim();
    return { thinking, text };
  }

  return { thinking: '', text: content };
}

function CodeBlock({ _node, inline, className, children, ...props }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const codeString = String(children).replace(/\n$/, '');

  if (inline) {
    return (
      <code style={{
        background: 'rgba(0, 0, 0, 0.4)',
        padding: '0.15rem 0.4rem',
        borderRadius: '4px',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.88em',
        color: '#f472b6'
      }} {...props}>
        {children}
      </code>
    );
  }

  const handleCopyCode = () => {
    navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{
      margin: '0.75rem 0',
      borderRadius: '8px',
      overflow: 'hidden',
      border: '1px solid var(--border-color)',
      background: 'rgba(15, 23, 42, 0.95)'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '0.4rem 0.85rem',
        background: 'rgba(30, 41, 59, 0.85)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        fontSize: '0.78rem',
        color: 'var(--text-muted)'
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-purple)' }}>
          {match ? match[1] : 'code'}
        </span>
        <button
          onClick={handleCopyCode}
          aria-label={t('chat.copyCode')}
          style={{
            background: 'none',
            border: 'none',
            color: copied ? 'var(--accent-green)' : 'var(--text-muted)',
            cursor: 'pointer',
            fontSize: '0.78rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem'
          }}
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          <span>{copied ? t('chat.copied') : t('chat.copy')}</span>
        </button>
      </div>
      <pre style={{
        margin: 0,
        padding: '0.85rem 1rem',
        overflowX: 'auto',
        fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
        fontSize: '0.85rem',
        lineHeight: '1.5',
        color: '#e2e8f0'
      }}>
        <code>{codeString}</code>
      </pre>
    </div>
  );
}

export default function ChatPlayground({
  apiBase,
  status,
  activeModel,
  isServerRunning,
  models = [],
  modelsLoading = false,
  onNavigateToControl,
  onNavigateToChat,
  onOpenShortcutsHelp
}) {
  const { t } = useTranslation();
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvIdState] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [expandedThinking, setExpandedThinking] = useState({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [generationError, setGenerationError] = useState(null);
  const [lastUserPrompt, setLastUserPrompt] = useState(null);

  // File upload & Audio recording state
  const [attachments, setAttachments] = useState([]);
  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [acceptFilter, setAcceptFilter] = useState('*/*');
  const [isDragging, setIsDragging] = useState(false);
  const [previewImageModal, setPreviewImageModal] = useState(null);

  // Audio recording state
  const [isVoiceRecording, setIsVoiceRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const recordingStreamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);

  const fileInputRef = useRef(null);
  const uploadMenuRef = useRef(null);

  // Generation parameters state
  const [genSettings, setGenSettings] = useState(() => getGlobalSettings());
  const [showSettings, setShowSettings] = useState(false);

  const [metrics, setMetrics] = useState({ tokens: 0, elapsed: 0, tps: 0, isApprox: false });
  const firstTokenTimeRef = useRef(null);
  const serverTpsRef = useRef(0);
  const tpsCalcRef = useRef(new SlidingWindowTpsCalculator(2000));

  // Poll status endpoint during generation to align tok/s with backend & BenchmarksView
  useEffect(() => {
    if (!isGenerating) {
      serverTpsRef.current = 0;
      return;
    }
    const fetchStatusMetrics = async () => {
      try {
        const res = await fetch(`${apiBase}/api/status`);
        if (res.ok) {
          const data = await res.json();
          if (data.tok_per_sec !== undefined && Number(data.tok_per_sec) > 0) {
            const val = Number(data.tok_per_sec);
            serverTpsRef.current = val;
            setMetrics(prev => ({
              ...prev,
              tps: val,
              isApprox: false
            }));
          }
        }
      } catch {
        // ignore status polling errors during streaming
      }
    };
    fetchStatusMetrics();
    const interval = setInterval(fetchStatusMetrics, 500);
    return () => clearInterval(interval);
  }, [isGenerating, apiBase]);

  useEffect(() => {
    if (status?.tok_per_sec !== undefined && Number(status.tok_per_sec) > 0) {
      const val = Number(status.tok_per_sec);
      serverTpsRef.current = val;
      if (isGenerating) {
        setMetrics(prev => ({
          ...prev,
          tps: val,
          isApprox: false
        }));
      }
    }
  }, [status, isGenerating]);

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Global app keyboard shortcuts: Esc, Ctrl+K, Ctrl+/, ?
  useKeyboardShortcuts({
    onNewConversation: () => {
      onNavigateToChat?.();
      handleNewConversation();
    },
    onToggleShortcutsHelp: () => {
      onOpenShortcutsHelp?.();
    },
    onEscape: () => {
      if (isGenerating) {
        handleStop();
      } else if (previewImageModal) {
        setPreviewImageModal(null);
      } else if (showSettings) {
        setShowSettings(false);
      } else if (showUploadMenu) {
        setShowUploadMenu(false);
      }
    }
  });

  // Helper to load settings from conversation or global default
  const applyConvSettings = (conv) => {
    const globalDefaults = getGlobalSettings();
    if (conv && conv.settings) {
      setGenSettings({ ...globalDefaults, ...conv.settings });
    } else if (conv && conv.systemPrompt !== undefined) {
      setGenSettings({ ...globalDefaults, systemPrompt: conv.systemPrompt });
    } else {
      setGenSettings(globalDefaults);
    }
  };

  // Load conversations on component mount
  useEffect(() => {
    const list = getConversations();
    setConversations(list);

    const activeId = getActiveConversationId();
    if (activeId) {
      const conv = list.find(c => c.id === activeId);
      if (conv) {
        setActiveConvIdState(conv.id);
        setMessages(conv.messages || []);
        applyConvSettings(conv);
        return;
      }
    }

    if (list.length > 0) {
      setActiveConvIdState(list[0].id);
      setMessages(list[0].messages || []);
      applyConvSettings(list[0]);
      setActiveConversationId(list[0].id);
    } else {
      const globalDefaults = getGlobalSettings();
      const welcomeConv = createConversation(
        t('sidebar.defaultTitle'),
        [{ role: 'assistant', content: t('chat.welcomeMsgContent') }],
        globalDefaults.systemPrompt,
        globalDefaults
      );
      setConversations([welcomeConv]);
      setActiveConvIdState(welcomeConv.id);
      setMessages(welcomeConv.messages);
      applyConvSettings(welcomeConv);
    }
  }, [t]);


  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (uploadMenuRef.current && !uploadMenuRef.current.contains(e.target)) {
        setShowUploadMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleThinking = (idx) => {
    setExpandedThinking(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const handleUpdateSettings = (newSettings) => {
    setGenSettings(newSettings);
    if (activeConvId) {
      const conv = getConversation(activeConvId);
      if (conv) {
        conv.settings = newSettings;
        conv.systemPrompt = newSettings.systemPrompt;
        saveConversation(conv);
        setConversations(getConversations());
      }
    }
  };

  const handleSaveAsGlobalDefaults = (settingsToSave) => {
    saveGlobalSettings(settingsToSave);
  };

  const handleResetToDefaults = () => {
    const globalDefaults = getGlobalSettings();
    handleUpdateSettings(globalDefaults);
  };

  const handleSelectConversation = (id) => {
    if (isGenerating) {
      if (!window.confirm(t('chat.confirmSwitchConv'))) {
        return;
      }
      handleStop();
    }
    const conv = getConversation(id);
    if (conv) {
      setActiveConvIdState(conv.id);
      setActiveConversationId(conv.id);
      setMessages(conv.messages || []);
      applyConvSettings(conv);
      setMetrics({ tokens: 0, elapsed: 0, tps: 0, isApprox: false });
      setExpandedThinking({});
      setAttachments([]);
      setGenerationError(null);
    }
  };

  const handleNewConversation = () => {
    if (isGenerating) {
      if (!window.confirm(t('chat.confirmNewChat'))) {
        return;
      }
      handleStop();
    }
    const currentGlobal = getGlobalSettings();
    const newConv = createConversation(t('sidebar.defaultTitle'), [], currentGlobal.systemPrompt, currentGlobal);
    setConversations(getConversations());
    setActiveConvIdState(newConv.id);
    setMessages([]);
    setGenSettings(currentGlobal);
    setMetrics({ tokens: 0, elapsed: 0, tps: 0, isApprox: false });
    setExpandedThinking({});
    setAttachments([]);
    setGenerationError(null);
  };

  const handleRenameConversation = (id, newTitle) => {
    renameConversation(id, newTitle);
    const updatedList = getConversations();
    setConversations(updatedList);
  };

  const handleDeleteConversation = (id) => {
    if (isGenerating && activeConvId === id) {
      handleStop();
    }
    deleteConversation(id);
    const updatedList = getConversations();
    setConversations(updatedList);

    if (activeConvId === id) {
      if (updatedList.length > 0) {
        const nextConv = updatedList[0];
        setActiveConvIdState(nextConv.id);
        setActiveConversationId(nextConv.id);
        setMessages(nextConv.messages || []);
        applyConvSettings(nextConv);
      } else {
        setActiveConvIdState(null);
        setActiveConversationId(null);
        setMessages([]);
      }
    }
  };

  const handleClearAllConversations = () => {
    if (isGenerating) handleStop();
    clearAllConversations();
    setConversations([]);
    setActiveConvIdState(null);
    setMessages([]);
    setMetrics({ tokens: 0, elapsed: 0, tps: 0, isApprox: false });
    setExpandedThinking({});
    setAttachments([]);
    setGenerationError(null);
  };

  const handleImportConversations = (updatedList) => {
    setConversations(updatedList);
    if (updatedList.length > 0 && !activeConvId) {
      const first = updatedList[0];
      setActiveConvIdState(first.id);
      setActiveConversationId(first.id);
      setMessages(first.messages || []);
      applyConvSettings(first);
    }
  };

  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef(null);

  const toggleRecording = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setGenerationError(t('chat.speechNotSupported') || 'Riconoscimento vocale non supportato dal browser.');
      return;
    }

    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = navigator.language || 'it-IT';

      recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript;
          }
        }
        if (transcript) {
          setInput(prev => (prev ? prev + ' ' : '') + transcript.trim());
        }
      };

      recognition.onerror = (e) => {
        console.error("Speech recognition error:", e);
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognition.start();
      recognitionRef.current = recognition;
      setIsRecording(true);
    } catch (e) {
      console.error("Failed to start speech recognition:", e);
      setIsRecording(false);
    }
  };

  const formatAudioDuration = (sec) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  const encodePCMToWAV = (samples, sampleRate = 16000) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono channel
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate (16-bit mono)
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  };

  const convertAudioBlobTo16kWav = async (fileOrBlob) => {
    const arrayBuffer = await fileOrBlob.arrayBuffer();
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtxClass();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    
    // Mixdown all channels to mono at 16kHz
    const srcChannels = decoded.numberOfChannels;
    const srcLen = decoded.length;
    const srcRate = decoded.sampleRate;
    const targetLen = Math.floor(srcLen * (16000 / srcRate));
    
    const monoSrc = new Float32Array(srcLen);
    for (let c = 0; c < srcChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < srcLen; i++) {
        monoSrc[i] += ch[i] / srcChannels;
      }
    }

    // Linear resample to 16kHz
    const resampled = new Float32Array(targetLen);
    const ratio = srcRate / 16000;
    for (let i = 0; i < targetLen; i++) {
      const srcIdx = i * ratio;
      const idx0 = Math.floor(srcIdx);
      const idx1 = Math.min(idx0 + 1, srcLen - 1);
      const frac = srcIdx - idx0;
      resampled[i] = monoSrc[idx0] * (1 - frac) + monoSrc[idx1] * frac;
    }

    await audioCtx.close();
    return encodePCMToWAV(resampled, 16000);
  };

  const audioContextRef = useRef(null);
  const audioProcessorRef = useRef(null);

  const handleStartVoiceRecording = async () => {
    setShowUploadMenu(false);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Il tuo browser non supporta la registrazione audio da microfono.");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingStreamRef.current = stream;
      audioChunksRef.current = [];

      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtxClass();
      audioContextRef.current = audioCtx;
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      audioProcessorRef.current = processor;
      window._alveareActiveAudioProc = processor; // Prevent V8 garbage collection

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setIsVoiceRecording(true);
      setRecordingTime(0);

      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1);
      }, 1000);

    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Impossibile accedere al microfono: " + (err.message || err));
      setIsVoiceRecording(false);
    }
  };

  const handleStopVoiceRecording = async () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }

    if (audioProcessorRef.current) {
      audioProcessorRef.current.disconnect();
      audioProcessorRef.current = null;
    }
    if (audioContextRef.current) {
      const srcRate = audioContextRef.current.sampleRate || 44100;
      await audioContextRef.current.close();
      audioContextRef.current = null;

      // Merge chunks
      let totalSamples = 0;
      for (const c of audioChunksRef.current) totalSamples += c.length;
      const rawMono = new Float32Array(totalSamples);
      let offset = 0;
      for (const c of audioChunksRef.current) {
        rawMono.set(c, offset);
        offset += c.length;
      }

      // Resample to 16,000 Hz if needed
      let finalSamples = rawMono;
      if (srcRate !== 16000 && rawMono.length > 0) {
        const targetLen = Math.floor(rawMono.length * (16000 / srcRate));
        finalSamples = new Float32Array(targetLen);
        const ratio = srcRate / 16000;
        for (let i = 0; i < targetLen; i++) {
          const srcIdx = i * ratio;
          const idx0 = Math.floor(srcIdx);
          const idx1 = Math.min(idx0 + 1, rawMono.length - 1);
          const frac = srcIdx - idx0;
          finalSamples[i] = rawMono[idx0] * (1 - frac) + rawMono[idx1] * frac;
        }
      }

      if (finalSamples.length > 0) {
        const wavBlob = encodePCMToWAV(finalSamples, 16000);
        const reader = new FileReader();
        reader.onload = () => {
          const dataUrl = reader.result;
          const timestampStr = new Date().toISOString().replace(/[:.]/g, '-').slice(11, 19);
          const attId = `rec_${Date.now()}`;
          const newAtt = {
            id: attId,
            name: `Registrazione_${timestampStr}.wav`,
            size: wavBlob.size,
            type: 'audio/wav',
            category: 'audio',
            url: dataUrl
          };
          setAttachments(prev => [...prev, newAtt]);
        };
        reader.readAsDataURL(wavBlob);
      }
    }

    if (recordingStreamRef.current) {
      recordingStreamRef.current.getTracks().forEach(track => track.stop());
      recordingStreamRef.current = null;
    }
    audioChunksRef.current = [];
    setIsVoiceRecording(false);
  };

  const handleCancelVoiceRecording = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (audioProcessorRef.current) {
      audioProcessorRef.current.disconnect();
      audioProcessorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (recordingStreamRef.current) {
      recordingStreamRef.current.getTracks().forEach(track => track.stop());
      recordingStreamRef.current = null;
    }
    audioChunksRef.current = [];
    setIsVoiceRecording(false);
  };

  const handleOpenUpload = (filterType) => {
    let accept = '*/*';
    if (filterType === 'image') accept = 'image/*,.png,.jpg,.jpeg,.webp,.gif,.bmp,.svg';
    else if (filterType === 'audio') accept = 'audio/*,.mp3,.wav,.ogg,.flac,.m4a,.aac,.webm';
    else if (filterType === 'document') accept = '.pdf,.docx,.txt,.md,.csv,.json,.py,.js,.ts,.jsx,.tsx,.html,.css,.cpp,.h,.c,.rs,.go,.yaml,.yml,.sh,.log,.sql,.xml,.epub,.rtf';
    
    setAcceptFilter(accept);
    setShowUploadMenu(false);
    setTimeout(() => {
      fileInputRef.current?.click();
    }, 50);
  };

  const handleFilesSelected = async (fileList) => {
    if (!fileList || fileList.length === 0) return;

    const newAttachments = [];

    for (const file of Array.from(fileList)) {
      const fileId = `att_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
      let category = 'document';
      if (file.type.startsWith('image/') || file.name.match(/\.(png|jpg|jpeg|webp|gif|bmp|svg)$/i)) category = 'image';
      else if (file.type.startsWith('audio/') || file.name.match(/\.(mp3|wav|ogg|flac|m4a|aac|webm)$/i)) category = 'audio';

      const item = {
        id: fileId,
        name: file.name,
        size: file.size,
        type: file.type || 'application/octet-stream',
        category,
        url: null,
        textData: null
      };

      if (category === 'image') {
        try {
          const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
          item.url = dataUrl;
        } catch (e) {
          console.error("Error reading image file:", e);
        }
      } else if (category === 'audio') {
        try {
          const wavBlob = await convertAudioBlobTo16kWav(file);
          const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(wavBlob);
          });
          item.url = dataUrl;
          item.type = 'audio/wav';
          item.size = wavBlob.size;
        } catch (e) {
          console.error("Error transcoding audio to WAV, fallback to raw read:", e);
          const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
          item.url = dataUrl;
        }
      } else {
        try {
          const content = await extractDocumentText(file);
          item.textData = content;
        } catch (e) {
          console.error("Error extracting document text:", e);
          item.textData = `[Errore lettura documento ${file.name}]`;
        }
      }

      newAttachments.push(item);
    }

    setAttachments(prev => [...prev, ...newAttachments]);
  };

  const removeAttachment = (id) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.relatedTarget === null || e.currentTarget.contains(e.relatedTarget)) {
      return;
    }
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesSelected(e.dataTransfer.files);
    }
  };

  const handleSend = async (overridePrompt) => {
    const messageText = overridePrompt !== undefined ? overridePrompt : input;
    if ((!messageText || !messageText.trim()) && attachments.length === 0) return;
    if (!isServerRunning) return;

    setGenerationError(null);
    setLastUserPrompt(messageText);

    let fullPromptContent = messageText.trim();
    let displayContent = messageText.trim();

    if (attachments.length > 0) {
      const docSummary = attachments
        .filter(a => a.textData)
        .map(a => `\n\n--- [Allegato: ${a.name}] ---\n${a.textData}\n--- [Fine ${a.name}] ---`)
        .join('');

      if (docSummary) {
        fullPromptContent = fullPromptContent ? `${fullPromptContent}\n${docSummary}` : docSummary.trim();
      }
    }

    const userMsg = {
      role: 'user',
      content: fullPromptContent,
      displayText: displayContent || undefined,
      attachments: [...attachments],
      timestamp: Date.now()
    };

    const updatedMessagesWithUser = [...messages, userMsg];
    let currentConvId = activeConvId;
    let currentTitle = 'Nuova conversazione';

    if (!currentConvId) {
      currentTitle = generateTitleFromMessage(userMsg.displayText || userMsg.content);
      const newConv = createConversation(currentTitle, updatedMessagesWithUser, genSettings.systemPrompt, genSettings);
      currentConvId = newConv.id;
      setActiveConvIdState(currentConvId);
    } else {
      const activeConv = getConversation(currentConvId);
      if (activeConv) {
        if (activeConv.title === 'Nuova conversazione' || activeConv.messages.length <= 1) {
          currentTitle = generateTitleFromMessage(userMsg.displayText || userMsg.content);
        } else {
          currentTitle = activeConv.title;
        }
      }
    }

    setMessages(updatedMessagesWithUser);
    setInput('');
    setAttachments([]);
    setIsGenerating(true);

    const assistantIndex = updatedMessagesWithUser.length;
    const initialAssistantMsg = { role: 'assistant', content: '', timestamp: Date.now() };
    const messagesForState = [...updatedMessagesWithUser, initialAssistantMsg];
    setMessages(messagesForState);

    saveConversation({
      id: currentConvId,
      title: currentTitle,
      messages: messagesForState,
      settings: genSettings,
      systemPrompt: genSettings.systemPrompt
    });
    setConversations(getConversations());

    const chatMessagesPayload = [];
    if (genSettings.systemPrompt && genSettings.systemPrompt.trim()) {
      chatMessagesPayload.push({ role: 'system', content: genSettings.systemPrompt.trim() });
    }
    for (const msg of updatedMessagesWithUser) {
      if (msg.role && msg.content !== undefined) {
        if (msg.attachments && msg.attachments.length > 0 && msg.role === 'user') {
          const parts = [];
          for (const att of msg.attachments) {
            if (att.category === 'image' && att.url) {
              parts.push({
                type: 'image_url',
                image_url: { url: att.url }
              });
            } else if (att.category === 'audio' && att.url) {
              parts.push({
                type: 'input_audio',
                input_audio: { data: att.url, format: 'wav' }
              });
            }
          }
          if (msg.displayText) {
            parts.push({ type: 'text', text: msg.displayText });
          } else if (msg.attachments.some(a => a.category === 'audio')) {
            parts.push({ type: 'text', text: 'Ascolta e trascrivi o rispondi a questo audio.' });
          }
          for (const att of msg.attachments) {
            if (att.textData) {
              parts.push({
                type: 'text',
                text: `\n\n--- [Allegato: ${att.name}] ---\n${att.textData}\n--- [Fine ${att.name}] ---`
              });
            }
          }
          chatMessagesPayload.push({
            role: msg.role,
            content: parts.length > 0 ? parts : msg.content
          });
        } else {
          chatMessagesPayload.push({
            role: msg.role,
            content: msg.content
          });
        }
      }
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    firstTokenTimeRef.current = null;
    serverTpsRef.current = 0;
    tpsCalcRef.current.reset();
    let accumulatedContent = '';

    try {
      const response = await fetch(`${apiBase}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel,
          messages: chatMessagesPayload,
          temperature: genSettings.temperature,
          top_p: genSettings.topP,
          top_k: genSettings.topK,
          max_tokens: genSettings.maxTokens,
          max_context_length: genSettings.maxContextLength,
          enable_thinking: genSettings.enableThinking,
          stream: true
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Errore Server (${response.status}): ${errText || 'Servizio momentaneamente non disponibile'}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue;
          if (trimmed === 'data: [DONE]') break;
          if (trimmed.startsWith('data: ')) {
            const jsonStr = trimmed.slice(6);
            try {
              const data = JSON.parse(jsonStr);
              const delta = data.choices?.[0]?.delta?.content || '';
              if (delta) {
                const now = Date.now();
                if (!firstTokenTimeRef.current) {
                  firstTokenTimeRef.current = now;
                }
                accumulatedContent += delta;
                
                setMessages(prev => {
                  const updated = [...prev];
                  if (updated[assistantIndex]) {
                    updated[assistantIndex] = {
                      ...updated[assistantIndex],
                      content: accumulatedContent
                    };
                  }
                  return updated;
                });

                const decodeElapsedSec = (now - firstTokenTimeRef.current) / 1000;
                const approxTokens = Math.max(1, Math.round(accumulatedContent.length / 4));

                tpsCalcRef.current.addSample(now, approxTokens);
                const clientTps = tpsCalcRef.current.getTps(now);

                const currentServerTps = serverTpsRef.current;
                const effectiveTps = currentServerTps > 0 ? currentServerTps : clientTps;
                const isApprox = currentServerTps <= 0;

                setMetrics({
                  tokens: approxTokens,
                  elapsed: Math.round(decodeElapsedSec * 10) / 10,
                  tps: effectiveTps,
                  isApprox
                });
              }
            } catch {
              // Ignore partial JSON parse errors
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        const friendlyErrorMsg = `Impossibile completare la risposta. ${err.message || 'Si è verificato un errore durante la connessione al server NPU.'}`;
        setGenerationError(friendlyErrorMsg);
        setMessages(prev => {
          const updated = [...prev];
          if (updated[assistantIndex]) {
            updated[assistantIndex] = {
              ...updated[assistantIndex],
              content: accumulatedContent ? `${accumulatedContent}\n\n⚠️ ${friendlyErrorMsg}` : `⚠️ ${friendlyErrorMsg}`
            };
          }
          return updated;
        });
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
      try {
        const res = await fetch(`${apiBase}/api/status`);
        if (res.ok) {
          const data = await res.json();
          if (data.tok_per_sec !== undefined && Number(data.tok_per_sec) > 0) {
            const finalTps = Number(data.tok_per_sec);
            serverTpsRef.current = finalTps;
            setMetrics(prev => ({ ...prev, tps: finalTps, isApprox: false }));
          }
        }
      } catch {
        // ignore final status fetch error
      }

      setMessages(latestMessages => {
        saveConversation({
          id: currentConvId,
          title: currentTitle,
          messages: latestMessages,
          settings: genSettings,
          systemPrompt: genSettings.systemPrompt
        });
        setConversations(getConversations());
        return latestMessages;
      });
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsGenerating(false);
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const activeConvObj = conversations.find(c => c.id === activeConvId);

  const isCustomizedSettings =
    genSettings.temperature !== DEFAULT_SETTINGS.temperature ||
    genSettings.topP !== DEFAULT_SETTINGS.topP ||
    genSettings.systemPrompt !== DEFAULT_SETTINGS.systemPrompt ||
    genSettings.enableThinking !== DEFAULT_SETTINGS.enableThinking ||
    genSettings.maxTokens !== DEFAULT_SETTINGS.maxTokens ||
    genSettings.maxContextLength !== DEFAULT_SETTINGS.maxContextLength;

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{ display: 'flex', height: 'calc(100vh - 64px)', width: '100%', overflow: 'hidden', position: 'relative' }}
      role="region"
      aria-label="Area Playground Chat Multimodale"
    >
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={e => handleFilesSelected(e.target.files)}
        style={{ display: 'none' }}
        accept={acceptFilter}
        multiple
        aria-hidden="true"
      />

      {/* Drag overlay */}
      {isDragging && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(15, 23, 42, 0.88)',
          backdropFilter: 'blur(8px)',
          zIndex: 500,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          border: '3px dashed var(--accent-purple)',
          margin: '0.75rem',
          borderRadius: '16px',
          pointerEvents: 'none'
        }}>
          <Upload size={54} color="var(--accent-purple)" className="pulse-icon" />
          <h3 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'white', marginTop: '1rem' }}>
            {t('chat.dragOverlayTitle')}
          </h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.3rem' }}>
            {t('chat.dragOverlayDesc')}
          </p>
        </div>
      )}

      {/* Conversation History Sidebar */}
      <SidebarHistory
        conversations={conversations}
        activeId={activeConvId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
        onClearAllConversations={handleClearAllConversations}
        onImportConversations={handleImportConversations}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Chat Workspace */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', position: 'relative', overflow: 'hidden' }}>
        
        {/* Top Chat Bar */}
        <div style={{
          padding: '0.85rem 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center',
          background: 'var(--bg-card)',
          flexWrap: 'wrap',
          gap: '0.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: 0 }}>
            <span style={{
              fontWeight: 700,
              fontSize: '1rem',
              color: 'var(--text-main)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '300px'
            }}>
              {activeConvObj?.title || t('chat.playgroundTitle')}
            </span>
            <span className="badge badge-success" style={{ textTransform: 'none', flexShrink: 0 }}>
              <Cpu size={14} /> {activeModel}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {metrics.tps > 0 && (
              <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                <Zap size={14} /> {t('chat.tokPerSec', { tps: metrics.isApprox ? `~${metrics.tps}` : metrics.tps, tokens: metrics.tokens, elapsed: metrics.elapsed })}
              </span>
            )}
            
            <button className="btn-secondary" onClick={handleNewConversation} title={t('sidebar.newChat')} aria-label={t('sidebar.newChat')}>
              <Plus size={16} /> {t('chat.newChatBtn')}
            </button>

            <button
              className={`btn-secondary ${showSettings ? 'active' : ''}`}
              onClick={() => setShowSettings(!showSettings)}
              title={t('chat.settingsTitle')}
              aria-label={t('chat.settingsTitle')}
              style={{ position: 'relative' }}
            >
              <Sliders size={16} /> {t('chat.settingsBtn')}
              {isCustomizedSettings && (
                <span
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: 'var(--accent-purple)',
                    display: 'inline-block',
                    marginLeft: '2px'
                  }}
                  title={t('chat.customSettingsActive')}
                />
              )}
            </button>
          </div>
        </div>

        {/* First-Run No Models Onboarding Banner */}
        {models.length === 0 && !modelsLoading && (
          <div style={{
            padding: '1rem 1.5rem',
            background: 'linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)',
            borderBottom: '1px solid rgba(245, 158, 11, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <HardDrive size={22} color="var(--accent-amber)" style={{ flexShrink: 0 }} />
              <div>
                <strong style={{ color: 'var(--text-main)', fontSize: '0.9rem' }}>
                  {t('chat.noModelsBannerTitle')}
                </strong>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  {t('chat.noModelsBannerDesc')}
                </div>
              </div>
            </div>

            <button
              className="btn-primary"
              onClick={onNavigateToControl}
              style={{ padding: '0.45rem 0.95rem', fontSize: '0.85rem' }}
              aria-label={t('chat.downloadModel')}
            >
              <Plus size={16} /> {t('chat.downloadModel')}
            </button>
          </div>
        )}

        {/* Message Stream or Empty Welcome Screen */}
        <div
          style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}
          role="log"
          aria-live="polite"
        >
          {messages.length === 0 ? (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              maxWidth: '680px',
              margin: '0 auto',
              textAlign: 'center',
              padding: '2rem 1rem',
              gap: '1.5rem'
            }}>
              <div style={{
                width: '64px',
                height: '64px',
                borderRadius: '16px',
                background: 'var(--gradient-brand)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: 'var(--shadow-glow)'
              }}>
                <Sparkles size={32} color="white" />
              </div>

              <div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.5rem' }}>
                  {t('chat.welcomeTitle')}
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: '1.6' }}>
                  {t('chat.welcomeDesc')}
                </p>
              </div>

              {/* Prompt Suggestions */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                gap: '0.85rem',
                width: '100%',
                marginTop: '1rem'
              }}>
                {[
                  { title: t('chat.promptSuggestions.npuTitle'), desc: t('chat.promptSuggestions.npuDesc') },
                  { title: t('chat.promptSuggestions.codeTitle'), desc: t('chat.promptSuggestions.codeDesc') },
                  { title: t('chat.promptSuggestions.archTitle'), desc: t('chat.promptSuggestions.archDesc') },
                  { title: t('chat.promptSuggestions.gemmaTitle'), desc: t('chat.promptSuggestions.gemmaDesc') }
                ].map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(prompt.desc)}
                    disabled={!isServerRunning || models.length === 0}
                    style={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '12px',
                      padding: '1rem',
                      textAlign: 'left',
                      cursor: isServerRunning && models.length > 0 ? 'pointer' : 'not-allowed',
                      transition: 'all 0.2s ease',
                      color: 'var(--text-main)',
                      opacity: isServerRunning && models.length > 0 ? 1 : 0.6
                    }}
                    className="glass-card"
                    aria-label={`Invia prompt: ${prompt.title}`}
                  >
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--accent-purple)', marginBottom: '0.25rem' }}>
                      {prompt.title}
                    </div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                      {prompt.desc}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => {
              const isUser = msg.role === 'user';
              const { thinking, text } = parseThinking(msg.content);
              const displayMessageText = msg.displayText !== undefined ? msg.displayText : text;
              const isCurrentlyGeneratingThis = isGenerating && idx === messages.length - 1;
              const isExpanded = !!expandedThinking[idx];

              return (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: isUser ? 'flex-end' : 'flex-start',
                    maxWidth: '850px',
                    alignSelf: isUser ? 'flex-end' : 'flex-start',
                    width: '100%'
                  }}
                >
                  <div style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.75rem',
                    flexDirection: isUser ? 'row-reverse' : 'row',
                    width: '100%'
                  }}>
                    {/* Avatar */}
                    <div style={{
                      width: '34px',
                      height: '34px',
                      borderRadius: '10px',
                      background: isUser ? 'var(--accent-purple)' : 'var(--gradient-brand)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                      boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                    }}>
                      {isUser ? <User size={18} color="white" /> : <Bot size={18} color="white" />}
                    </div>

                    {/* Message Bubble */}
                    <div style={{
                      flex: 1,
                      background: isUser ? 'rgba(139, 92, 246, 0.15)' : 'var(--bg-card)',
                      border: isUser ? '1px solid rgba(139, 92, 246, 0.3)' : '1px solid var(--border-color)',
                      borderRadius: '14px',
                      padding: '1rem 1.25rem',
                      color: 'var(--text-main)',
                      boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
                      position: 'relative'
                    }}>
                      {/* Attachments preview if user message */}
                      {msg.attachments && msg.attachments.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', marginBottom: '0.85rem' }}>
                          {msg.attachments.map(att => (
                            <div
                              key={att.id}
                              style={{
                                background: 'rgba(0,0,0,0.3)',
                                border: '1px solid var(--border-color)',
                                borderRadius: '8px',
                                padding: '0.4rem 0.6rem',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                fontSize: '0.8rem'
                              }}
                            >
                              {att.category === 'image' && att.url ? (
                                <img
                                  src={att.url}
                                  alt={att.name}
                                  onClick={() => setPreviewImageModal(att.url)}
                                  style={{ width: '48px', height: '48px', objectFit: 'cover', borderRadius: '6px', cursor: 'pointer' }}
                                />
                              ) : att.category === 'audio' && att.url ? (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                  <span style={{ fontWeight: 500, fontSize: '0.75rem', color: 'var(--text-muted)' }}>{att.name}</span>
                                  <audio src={att.url} controls style={{ height: '30px', maxWidth: '240px' }} />
                                </div>
                              ) : (
                                <>
                                  <FileText size={16} color="var(--accent-cyan)" />
                                  <span style={{ fontWeight: 500, maxWidth: '160px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {att.name}
                                  </span>
                                </>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Thinking Chain-of-Thought Section */}
                      {!isUser && thinking && (
                        <div style={{
                          marginBottom: '1rem',
                          background: 'rgba(139, 92, 246, 0.08)',
                          border: '1px solid rgba(139, 92, 246, 0.25)',
                          borderRadius: '10px',
                          overflow: 'hidden'
                        }}>
                          <button
                            onClick={() => toggleThinking(idx)}
                            aria-expanded={isExpanded}
                            style={{
                              width: '100%',
                              padding: '0.6rem 0.85rem',
                              background: 'none',
                              border: 'none',
                              color: 'var(--accent-purple)',
                              fontWeight: 600,
                              fontSize: '0.82rem',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between'
                            }}
                          >
                            <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                              <Brain size={16} /> {t('chat.thinkingTitle')}
                            </span>
                            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                          </button>

                          {(isExpanded || (isCurrentlyGeneratingThis && !displayMessageText)) && (
                            <div style={{
                              padding: '0.85rem 1rem',
                              borderTop: '1px solid rgba(139, 92, 246, 0.15)',
                              fontSize: '0.86rem',
                              lineHeight: '1.6',
                              color: 'var(--text-muted)',
                              whiteSpace: 'pre-wrap',
                              fontFamily: 'monospace'
                            }}>
                              {thinking}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Message Content */}
                      <div className="markdown-content" style={{ fontSize: '0.94rem', lineHeight: '1.65' }}>
                        {isCurrentlyGeneratingThis && !displayMessageText ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
                            <Zap size={16} className="spin-icon" color="var(--accent-cyan)" />
                            <span>{t('chat.initializingNpu')}</span>
                          </div>
                        ) : (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {displayMessageText}
                          </ReactMarkdown>
                        )}
                      </div>

                      {/* Actions footer */}
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: isUser ? 'flex-end' : 'space-between',
                        marginTop: '0.65rem',
                        gap: '0.5rem',
                        borderTop: isUser ? 'none' : '1px solid rgba(255,255,255,0.05)',
                        paddingTop: isUser ? 0 : '0.45rem'
                      }}>
                        {!isUser && (
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <Cpu size={12} color="var(--accent-cyan)" />
                            {activeModel}
                          </span>
                        )}

                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <button
                            onClick={() => handleCopy(displayMessageText || msg.content, idx)}
                            title={t('chat.copyResponse')}
                            aria-label={t('chat.copyResponse')}
                            style={{
                              background: 'none',
                              border: 'none',
                              color: copiedIndex === idx ? 'var(--accent-green)' : 'var(--text-muted)',
                              cursor: 'pointer',
                              padding: '0.2rem 0.4rem',
                              borderRadius: '4px',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.25rem',
                              fontSize: '0.72rem'
                            }}
                          >
                            {copiedIndex === idx ? <Check size={13} /> : <Copy size={13} />}
                            {copiedIndex === idx ? t('chat.copied') : t('chat.copy')}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Retry Banner after interruption */}
        {generationError && (
          <div style={{
            padding: '0.75rem 1.5rem',
            background: 'rgba(239, 68, 68, 0.1)',
            borderTop: '1px solid rgba(239, 68, 68, 0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            fontSize: '0.85rem',
            color: 'var(--accent-red)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertTriangle size={16} />
              <span>{generationError}</span>
            </div>
            {lastUserPrompt && (
              <button
                className="btn-secondary"
                onClick={() => {
                  handleSend(lastUserPrompt);
                }}
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              >
                <RefreshCw size={14} /> {t('chat.retry')}
              </button>
            )}
          </div>
        )}

        {/* Active Attachments Bar */}
        {attachments.length > 0 && (
          <div style={{
            padding: '0.5rem 1.5rem',
            background: 'rgba(0,0,0,0.2)',
            borderTop: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            overflowX: 'auto'
          }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, flexShrink: 0 }}>
              {t('chat.attachmentsLabel', { count: attachments.length })}
            </span>
            {attachments.map(att => (
              <div
                key={att.id}
                style={{
                  background: 'rgba(139, 92, 246, 0.15)',
                  border: '1px solid rgba(139, 92, 246, 0.3)',
                  borderRadius: '6px',
                  padding: '0.25rem 0.55rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  fontSize: '0.78rem',
                  color: 'white',
                  flexShrink: 0
                }}
              >
                {att.category === 'image' && att.url ? (
                  <img
                    src={att.url}
                    alt={att.name}
                    style={{ width: '18px', height: '18px', objectFit: 'cover', borderRadius: '3px' }}
                  />
                ) : att.category === 'audio' ? (
                  <>
                    <Music size={14} color="var(--accent-cyan)" />
                    {att.url && (
                      <button
                        onClick={() => {
                          const a = new Audio(att.url);
                          a.play().catch(e => console.error("Audio play error:", e));
                        }}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer', padding: '0 2px', display: 'flex', alignItems: 'center' }}
                        title="Ascolta anteprima registrata"
                        aria-label="Ascolta anteprima"
                      >
                        <Play size={11} />
                      </button>
                    )}
                  </>
                ) : (
                  <FileText size={14} color="var(--accent-green)" />
                )}
                <span style={{ maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {att.name}
                </span>
                <button
                  onClick={() => removeAttachment(att.id)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 0 }}
                  aria-label={`Rimuovi allegato ${att.name}`}
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input Bar */}
        <div style={{
          padding: '0.85rem 1.5rem',
          borderTop: '1px solid var(--border-color)',
          background: 'var(--bg-secondary)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', position: 'relative' }}>
            
            {/* Attachment Button */}
            <div style={{ position: 'relative' }} ref={uploadMenuRef}>
              <button
                className="btn-secondary"
                onClick={() => setShowUploadMenu(!showUploadMenu)}
                title={t('chat.attachFile')}
                aria-label={t('chat.attachFile')}
                aria-expanded={showUploadMenu}
                disabled={!isServerRunning || isGenerating}
                style={{ padding: '0.75rem', borderRadius: '10px' }}
              >
                <Paperclip size={18} />
              </button>

              {/* Upload Dropdown Menu */}
              {showUploadMenu && (
                <div
                  role="menu"
                  style={{
                    position: 'absolute',
                    bottom: '100%',
                    left: 0,
                    marginBottom: '0.5rem',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '10px',
                    padding: '0.4rem',
                    boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.2rem',
                    zIndex: 100,
                    width: '190px'
                  }}
                >
                  <button
                    onClick={() => handleOpenUpload('image')}
                    role="menuitem"
                    style={{ background: 'none', border: 'none', color: 'var(--text-main)', padding: '0.5rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', textAlign: 'left', width: '100%' }}
                  >
                    <ImageIcon size={16} color="var(--accent-purple)" /> {t('chat.image')}
                  </button>
                  <button
                    onClick={() => handleOpenUpload('audio')}
                    role="menuitem"
                    style={{ background: 'none', border: 'none', color: 'var(--text-main)', padding: '0.5rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', textAlign: 'left', width: '100%' }}
                  >
                    <Music size={16} color="var(--accent-cyan)" /> {t('chat.audio')}
                  </button>
                  <button
                    onClick={handleStartVoiceRecording}
                    role="menuitem"
                    style={{ background: 'none', border: 'none', color: 'var(--text-main)', padding: '0.5rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', textAlign: 'left', width: '100%' }}
                  >
                    <Mic size={16} color="#ef4444" /> Registra vocale
                  </button>
                  <button
                    onClick={() => handleOpenUpload('document')}
                    role="menuitem"
                    style={{ background: 'none', border: 'none', color: 'var(--text-main)', padding: '0.5rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', textAlign: 'left', width: '100%' }}
                  >
                    <FileText size={16} color="var(--accent-green)" /> {t('chat.document')}
                  </button>
                </div>
              )}
            </div>

            {/* Live Audio Recording Bar or Text Input */}
            {isVoiceRecording ? (
              <div style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid rgba(239, 68, 68, 0.35)',
                borderRadius: '10px',
                padding: '0.5rem 1rem',
                height: '52px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: '#ef4444',
                    boxShadow: '0 0 12px #ef4444'
                  }} className="pulse-icon" />
                  <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f87171' }}>
                    Registrazione vocale in corso... {formatAudioDuration(recordingTime)}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <button
                    onClick={handleCancelVoiceRecording}
                    className="btn-secondary"
                    style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}
                  >
                    Annulla
                  </button>
                  <button
                    onClick={handleStopVoiceRecording}
                    style={{
                      background: '#ef4444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '0.4rem 0.85rem',
                      fontSize: '0.82rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem'
                    }}
                  >
                    <StopCircle size={15} /> Ferma e Allega
                  </button>
                </div>
              </div>
            ) : (
              <>
                {/* Microphone Button (Speech to Text) */}
                <button
                  className={isRecording ? "btn-danger pulse-icon" : "btn-secondary"}
                  onClick={toggleRecording}
                  title={isRecording ? "Ferma trascrizione vocale" : "Trascrizione vocale (Speech-to-Text)"}
                  aria-label={isRecording ? "Ferma trascrizione vocale" : "Trascrizione vocale"}
                  disabled={!isServerRunning || isGenerating}
                  style={{
                    padding: '0.75rem',
                    borderRadius: '10px',
                    background: isRecording ? 'rgba(239, 68, 68, 0.25)' : undefined,
                    borderColor: isRecording ? 'rgba(239, 68, 68, 0.6)' : undefined,
                    color: isRecording ? 'var(--accent-red, #ef4444)' : undefined
                  }}
                >
                  {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
                </button>

                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      if (e.nativeEvent.isComposing) return;
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder={
                    !isServerRunning
                      ? t('chat.inputPlaceholderServerOff')
                      : models.length === 0
                      ? t('chat.inputPlaceholderNoModels')
                      : t('chat.inputPlaceholderReady')
                  }
                  disabled={!isServerRunning || isGenerating || models.length === 0}
                  aria-label="Campo testo messaggio"
                  style={{
                    flex: 1,
                    padding: '0.8rem 1rem',
                    borderRadius: '10px',
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-main)',
                    fontSize: '0.95rem',
                    resize: 'none',
                    height: '52px',
                    outline: 'none',
                    fontFamily: 'var(--font-sans)'
                  }}
                />
              </>
            )}

            {isGenerating ? (
              <button
                className="btn-danger"
                onClick={handleStop}
                style={{ height: '52px' }}
                aria-label={t('chat.stop')}
                title={t('chat.stopTitle')}
              >
                <StopCircle size={20} /> {t('chat.stop')}
              </button>
            ) : (
              <button
                className="btn-primary"
                onClick={() => handleSend()}
                disabled={!isServerRunning || (!input.trim() && attachments.length === 0) || models.length === 0}
                style={{ height: '52px' }}
                aria-label={t('chat.send')}
              >
                <Send size={18} /> {t('chat.send')}
              </button>
            )}
          </div>

          {/* Keyboard shortcut legend footer */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.4rem', fontSize: '0.72rem', color: 'var(--text-subtle)' }}>
            <span>{t('chat.shortcutNoticeDrag')}</span>
            <button
              onClick={onOpenShortcutsHelp}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '0.72rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: 0
              }}
              title={t('shortcuts.buttonTitle')}
            >
              <span>{t('chat.shortcutNoticeKeys')}</span>
              <kbd style={{
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                padding: '0.1rem 0.35rem',
                fontSize: '0.68rem',
                fontFamily: 'var(--font-mono)'
              }}>Ctrl+/</kbd>
            </button>
          </div>
        </div>
      </div>

      {/* Image Preview Modal */}
      {previewImageModal && (
        <div
          onClick={() => setPreviewImageModal(null)}
          role="dialog"
          aria-modal="true"
          aria-label="Anteprima Immagine"
          tabIndex={-1}
          onKeyDown={e => {
            if (e.key === 'Escape') setPreviewImageModal(null);
          }}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.85)',
            backdropFilter: 'blur(5px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '2rem'
          }}
        >
          <img
            src={previewImageModal}
            alt="Anteprima immagine in dimensioni reali"
            style={{ maxWidth: '90%', maxHeight: '90%', borderRadius: '12px', boxShadow: '0 0 40px rgba(0,0,0,0.8)' }}
          />
        </div>
      )}

      {/* Generation Settings Drawer/Modal */}
      <GenerationSettingsPanel
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={genSettings}
        onUpdateSettings={handleUpdateSettings}
        onSaveAsGlobalDefaults={handleSaveAsGlobalDefaults}
        onResetToDefaults={handleResetToDefaults}
      />
    </div>
  );
}
