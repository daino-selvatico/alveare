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
  HardDrive
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
  activeModel,
  isServerRunning,
  models = [],
  modelsLoading = false,
  onNavigateToControl
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

  // File upload state
  const [attachments, setAttachments] = useState([]);
  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [acceptFilter, setAcceptFilter] = useState('*/*');
  const [isDragging, setIsDragging] = useState(false);
  const [previewImageModal, setPreviewImageModal] = useState(null);

  const fileInputRef = useRef(null);
  const uploadMenuRef = useRef(null);

  // Generation parameters state
  const [genSettings, setGenSettings] = useState(() => getGlobalSettings());
  const [showSettings, setShowSettings] = useState(false);

  const [metrics, setMetrics] = useState({ tokens: 0, elapsed: 0, tps: 0 });

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Keyboard shortcut listener: Esc to stop generation or close modals
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
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
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isGenerating, previewImageModal, showSettings, showUploadMenu]);

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
      setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
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
    setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
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
    setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
    setExpandedThinking({});
    setAttachments([]);
    setGenerationError(null);
  };

  const handleOpenUpload = (filterType) => {
    let accept = '*/*';
    if (filterType === 'image') accept = 'image/*';
    else if (filterType === 'audio') accept = 'audio/*';
    else if (filterType === 'document') accept = '.pdf,.txt,.md,.csv,.json,.py,.js,.ts,.jsx,.tsx,.html,.css,.cpp,.h,.c,.rs,.go,.yaml,.yml,.sh,.log';
    
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
      if (file.type.startsWith('image/')) category = 'image';
      else if (file.type.startsWith('audio/')) category = 'audio';

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
        item.url = URL.createObjectURL(file);
      }

      if (category === 'document' || file.name.match(/\.(txt|md|csv|json|py|js|ts|jsx|tsx|html|css|cpp|h|c|rs|go|yaml|yml|sh|log)$/i)) {
        try {
          const content = await file.text();
          item.textData = content;
        } catch {
          // Ignore binary reading error
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
      const attachSummary = attachments.map(a => {
        if (a.textData) {
          return `\n\n--- [Allegato: ${a.name}] ---\n${a.textData}\n--- [Fine ${a.name}] ---`;
        }
        return `\n[Allegato caricato: ${a.name} (${a.category}, ${(a.size / 1024).toFixed(1)} KB)]`;
      }).join('');

      fullPromptContent = fullPromptContent ? `${fullPromptContent}\n${attachSummary}` : attachSummary.trim();
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
        chatMessagesPayload.push({
          role: msg.role,
          content: msg.content
        });
      }
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const startTime = Date.now();
    let tokenCount = 0;
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
                tokenCount++;
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

                const elapsedSec = (Date.now() - startTime) / 1000;
                setMetrics({
                  tokens: tokenCount,
                  elapsed: Math.round(elapsedSec * 10) / 10,
                  tps: elapsedSec > 0 ? Math.round((tokenCount / elapsedSec) * 10) / 10 : 0
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
                <Zap size={14} /> {t('chat.tokPerSec', { tps: metrics.tps, tokens: metrics.tokens, elapsed: metrics.elapsed })}
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
                                  style={{ width: '40px', height: '40px', objectFit: 'cover', borderRadius: '6px', cursor: 'pointer' }}
                                />
                              ) : (
                                <FileText size={16} color="var(--accent-cyan)" />
                              )}
                              <span style={{ fontWeight: 500, maxWidth: '160px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {att.name}
                              </span>
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
                              {isCurrentlyGeneratingThis && !displayMessageText && (
                                <span className="pulse-icon" style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)' }}>
                                  {t('chat.thinkingProcessing')}
                                </span>
                              )}
                            </span>
                            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                          </button>

                          {(isExpanded || (isCurrentlyGeneratingThis && !displayMessageText)) && (
                            <div style={{
                              padding: '0.75rem 0.85rem',
                              borderTop: '1px solid rgba(139, 92, 246, 0.15)',
                              fontSize: '0.84rem',
                              lineHeight: '1.5',
                              color: 'var(--text-muted)',
                              fontFamily: 'var(--font-mono)',
                              whiteSpace: 'pre-wrap',
                              background: 'rgba(0,0,0,0.15)'
                            }}>
                              {thinking}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Streaming First-Token Pulse Placeholder */}
                      {isCurrentlyGeneratingThis && !displayMessageText && !thinking && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--accent-cyan)', fontSize: '0.9rem', padding: '0.2rem 0' }}>
                          <Sparkles size={18} className="pulse-icon" />
                          <span>{t('chat.initializingNpu')}</span>
                        </div>
                      )}

                      {/* Markdown Text Body */}
                      <div style={{ fontSize: '0.94rem', lineHeight: '1.6' }}>
                        {isUser ? (
                          <div style={{ whiteSpace: 'pre-wrap' }}>{displayMessageText}</div>
                        ) : (
                          displayMessageText && (
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                code: CodeBlock
                              }}
                            >
                              {displayMessageText}
                            </ReactMarkdown>
                          )
                        )}
                      </div>

                      {/* Copy Action */}
                      {!isUser && displayMessageText && (
                        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.6rem' }}>
                          <button
                            onClick={() => copyToClipboard(displayMessageText, idx)}
                            aria-label={t('chat.copyResponse')}
                            style={{
                              background: 'none',
                              border: 'none',
                              color: copiedIndex === idx ? 'var(--accent-green)' : 'var(--text-muted)',
                              cursor: 'pointer',
                              fontSize: '0.76rem',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.3rem'
                            }}
                          >
                            {copiedIndex === idx ? <Check size={14} /> : <Copy size={14} />}
                            <span>{copiedIndex === idx ? t('chat.copied') : t('chat.copy')}</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          )}

          {/* Resilient Error Retry Banner for failed generation */}
          {generationError && lastUserPrompt && !isGenerating && (
            <div style={{
              alignSelf: 'center',
              maxWidth: '600px',
              width: '100%',
              margin: '0.5rem 0',
              padding: '0.85rem 1.25rem',
              borderRadius: '12px',
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: '#fca5a5', fontSize: '0.85rem' }}>
                <AlertTriangle size={18} style={{ flexShrink: 0 }} />
                <span>{t('chat.generationInterrupted')}</span>
              </div>
              <button
                className="btn-secondary"
                onClick={() => handleSend(lastUserPrompt)}
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', color: '#fca5a5', borderColor: 'rgba(239, 68, 68, 0.4)' }}
                aria-label={t('chat.retry')}
              >
                <RefreshCw size={14} /> {t('chat.retry')}
              </button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Selected Attachments Bar */}
        {attachments.length > 0 && (
          <div style={{
            padding: '0.6rem 1.5rem',
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
                <FileText size={14} color="var(--accent-purple)" />
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
                    width: '170px'
                  }}
                >
                  <button
                    onClick={() => handleOpenUpload('image')}
                    role="menuitem"
                    style={{ background: 'none', border: 'none', color: 'var(--text-main)', padding: '0.5rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', textAlign: 'left' }}
                  >
                    <ImageIcon size={16} color="var(--accent-purple)" /> {t('chat.image')}
                  </button>
                  <button
                    onClick={() => handleOpenUpload('audio')}
                    role="menuitem"
                    style={{ background: 'none', border: 'none', color: 'var(--text-main)', padding: '0.5rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', textAlign: 'left' }}
                  >
                    <Music size={16} color="var(--accent-cyan)" /> {t('chat.audio')}
                  </button>
                  <button
                    onClick={() => handleOpenUpload('document')}
                    role="menuitem"
                    style={{ background: 'none', border: 'none', color: 'var(--text-main)', padding: '0.5rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', textAlign: 'left' }}
                  >
                    <FileText size={16} color="var(--accent-green)" /> {t('chat.document')}
                  </button>
                </div>
              )}
            </div>

            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
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
            <span>{t('chat.shortcutNoticeKeys')}</span>
          </div>
        </div>
      </div>

      {/* Image Preview Modal */}
      {previewImageModal && (
        <div
          onClick={() => setPreviewImageModal(null)}
          role="dialog"
          aria-label="Anteprima Immagine"
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
            alt="Anteprima"
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
