import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Bot,
  User,
  Sliders,
  Cpu,
  Zap,
  StopCircle,
  Copy,
  Check,
  Brain,
  ChevronDown,
  ChevronUp,
  ToggleLeft,
  ToggleRight,
  Plus,
  Sparkles
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SidebarHistory from './SidebarHistory';
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
  generateTitleFromMessage
} from '../utils/chatStorage';

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

function CodeBlock({ inline, className, children, ...props }) {
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
          <span>{copied ? 'Copiato!' : 'Copia'}</span>
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

export default function ChatPlayground({ apiBase, activeModel, isServerRunning }) {
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvIdState] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [expandedThinking, setExpandedThinking] = useState({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Generation parameters
  const [systemPrompt, setSystemPrompt] = useState('Sei un assistente AI esperto ed utile.');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(512);
  const [maxContextLength, setMaxContextLength] = useState(4096);
  const [enableThinking, setEnableThinking] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  // Performance metrics
  const [metrics, setMetrics] = useState({ tokens: 0, elapsed: 0, tps: 0 });

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

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
        if (conv.systemPrompt !== undefined) {
          setSystemPrompt(conv.systemPrompt);
        }
        return;
      }
    }

    // If there are existing conversations, select the first one; otherwise start empty
    if (list.length > 0) {
      setActiveConvIdState(list[0].id);
      setMessages(list[0].messages || []);
      if (list[0].systemPrompt !== undefined) {
        setSystemPrompt(list[0].systemPrompt);
      }
      setActiveConversationId(list[0].id);
    } else {
      // Create initial welcome conversation
      const welcomeConv = createConversation(
        'Prima conversazione',
        [{ role: 'assistant', content: "Ciao! Sono il modello LLM in esecuzione sull'NPU AMD Ryzen AI. Come posso aiutarti oggi?" }],
        'Sei un assistente AI esperto ed utile.'
      );
      setConversations([welcomeConv]);
      setActiveConvIdState(welcomeConv.id);
      setMessages(welcomeConv.messages);
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleThinking = (idx) => {
    setExpandedThinking(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  // Switch to a selected conversation
  const handleSelectConversation = (id) => {
    if (isGenerating) {
      if (!window.confirm("C'è una generazione in corso. Vuoi interromperla per cambiare conversazione?")) {
        return;
      }
      handleStop();
    }
    const conv = getConversation(id);
    if (conv) {
      setActiveConvIdState(conv.id);
      setActiveConversationId(conv.id);
      setMessages(conv.messages || []);
      if (conv.systemPrompt !== undefined) {
        setSystemPrompt(conv.systemPrompt);
      }
      setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
      setExpandedThinking({});
    }
  };

  // Start a new empty conversation
  const handleNewConversation = () => {
    if (isGenerating) {
      if (!window.confirm("C'è una generazione in corso. Vuoi interromperla per creare una nuova chat?")) {
        return;
      }
      handleStop();
    }
    const newConv = createConversation('Nuova conversazione', [], systemPrompt);
    setConversations(getConversations());
    setActiveConvIdState(newConv.id);
    setMessages([]);
    setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
    setExpandedThinking({});
  };

  // Rename a conversation
  const handleRenameConversation = (id, newTitle) => {
    renameConversation(id, newTitle);
    const updatedList = getConversations();
    setConversations(updatedList);
  };

  // Delete a conversation
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
      } else {
        setActiveConvIdState(null);
        setActiveConversationId(null);
        setMessages([]);
      }
    }
  };

  // Clear all saved conversations
  const handleClearAllConversations = () => {
    if (isGenerating) handleStop();
    clearAllConversations();
    setConversations([]);
    setActiveConvIdState(null);
    setMessages([]);
    setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
  };

  // Send message and execute full multi-turn completion
  const handleSend = async (customInput) => {
    const textToSend = typeof customInput === 'string' ? customInput : input;
    if (!textToSend.trim() || isGenerating) return;
    if (!isServerRunning) {
      alert("Il server di inferenza non è in esecuzione! Avvialo dal Control Panel.");
      return;
    }

    const userMsg = { role: 'user', content: textToSend.trim(), timestamp: Date.now() };
    const updatedMessagesWithUser = [...messages, userMsg];
    
    // Manage conversation instance
    let currentConvId = activeConvId;
    let currentTitle = 'Nuova conversazione';

    if (!currentConvId) {
      currentTitle = generateTitleFromMessage(userMsg.content);
      const newConv = createConversation(currentTitle, updatedMessagesWithUser, systemPrompt);
      currentConvId = newConv.id;
      setActiveConvIdState(currentConvId);
    } else {
      const activeConv = getConversation(currentConvId);
      if (activeConv) {
        // Auto title if conversation currently has default title or only welcome message
        if (activeConv.title === 'Nuova conversazione' || activeConv.messages.length <= 1) {
          currentTitle = generateTitleFromMessage(userMsg.content);
        } else {
          currentTitle = activeConv.title;
        }
      }
    }

    setMessages(updatedMessagesWithUser);
    setInput('');
    setIsGenerating(true);

    const assistantIndex = updatedMessagesWithUser.length;
    const initialAssistantMsg = { role: 'assistant', content: '', timestamp: Date.now() };
    const messagesForState = [...updatedMessagesWithUser, initialAssistantMsg];
    setMessages(messagesForState);

    // Save state to localStorage
    saveConversation({
      id: currentConvId,
      title: currentTitle,
      messages: messagesForState,
      systemPrompt
    });
    setConversations(getConversations());

    // Prepare full multi-turn history payload for /v1/chat/completions
    const chatMessagesPayload = [];
    if (systemPrompt.trim()) {
      chatMessagesPayload.push({ role: 'system', content: systemPrompt.trim() });
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
          temperature,
          top_p: topP,
          top_k: topK,
          max_tokens: maxTokens,
          max_context_length: maxContextLength,
          enable_thinking: enableThinking,
          stream: true
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Errore Server (${response.status}): ${errText}`);
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
          if (trimmed === 'data: [DONE]') {
            break;
          }
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
        accumulatedContent += `\n\n⚠️ [Errore]: ${err.message}`;
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
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;

      // Final save to localStorage once turn completes
      setMessages(latestMessages => {
        saveConversation({
          id: currentConvId,
          title: currentTitle,
          messages: latestMessages,
          systemPrompt
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

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 64px)', width: '100%', overflow: 'hidden' }}>
      
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
          background: 'rgba(19, 27, 46, 0.5)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: 0 }}>
            <span style={{
              fontWeight: 700,
              fontSize: '1rem',
              color: 'white',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '300px'
            }}>
              {activeConvObj?.title || 'Playground Realtime'}
            </span>
            <span className="badge badge-success" style={{ textTransform: 'none', flexShrink: 0 }}>
              <Cpu size={14} /> {activeModel}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {metrics.tps > 0 && (
              <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                <Zap size={14} /> {metrics.tps} tok/s ({metrics.tokens} tok in {metrics.elapsed}s)
              </span>
            )}
            
            <button className="btn-secondary" onClick={handleNewConversation} title="Nuova Conversazione">
              <Plus size={16} /> Nuova Chat
            </button>

            <button className="btn-secondary" onClick={() => setShowSettings(!showSettings)} title="Parametri di generazione">
              <Sliders size={16} /> Impostazioni
            </button>
          </div>
        </div>

        {/* Message Stream or Welcome Screen */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
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
                boxShadow: '0 0 30px rgba(139, 92, 246, 0.4)'
              }}>
                <Sparkles size={32} color="white" />
              </div>

              <div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'white', marginBottom: '0.5rem' }}>
                  Alveare NPU Chat Playground
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: '1.6' }}>
                  Chatta direttamente in locale con i modelli LLM accelerati da hardware AMD Ryzen AI (XDNA2 NPU).
                  I turni della conversazione vengono memorizzati con riutilizzo della cache KV.
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
                  { title: "Funzionamento NPU", desc: "Spiegami come l'NPU AMD Ryzen AI accelera l'inferenza LLM." },
                  { title: "Generazione Codice", desc: "Scrivi un server HTTP in Python ad alte prestazioni." },
                  { title: "Analisi Architetturale", desc: "Quali sono i vantaggi del KV-cache prefix reuse nell'NPU?" },
                  { title: "Caratteristiche Gemma 4", desc: "Cosa contraddistingue i modelli Gemma 4 e CoT (Chain-of-Thought)?" }
                ].map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(prompt.desc)}
                    style={{
                      background: 'rgba(30, 41, 59, 0.6)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '12px',
                      padding: '1rem',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      color: 'var(--text-main)'
                    }}
                    className="glass-card"
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
              const isCurrentlyGeneratingThis = isGenerating && idx === messages.length - 1;
              const isExpanded = !!expandedThinking[idx];

              return (
                <div key={idx} style={{
                  display: 'flex',
                  gap: '1rem',
                  maxWidth: '85%',
                  alignSelf: isUser ? 'flex-end' : 'flex-start',
                  flexDirection: isUser ? 'row-reverse' : 'row'
                }}>
                  <div style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    background: isUser ? 'var(--gradient-brand)' : 'rgba(139, 92, 246, 0.2)',
                    border: !isUser ? '1px solid var(--accent-purple)' : 'none',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}>
                    {isUser ? <User size={18} color="white" /> : <Bot size={18} color="#c084fc" />}
                  </div>

                  <div style={{
                    background: isUser ? 'rgba(139, 92, 246, 0.18)' : 'rgba(30, 41, 59, 0.7)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '12px',
                    padding: '0.85rem 1.1rem',
                    color: 'var(--text-main)',
                    fontSize: '0.95rem',
                    lineHeight: '1.6',
                    position: 'relative',
                    minWidth: '240px'
                  }}>
                    {/* Thinking Section */}
                    {!isUser && thinking && (
                      isCurrentlyGeneratingThis ? (
                        <div style={{
                          background: 'rgba(139, 92, 246, 0.12)',
                          border: '1px solid rgba(139, 92, 246, 0.3)',
                          borderRadius: '8px',
                          padding: '0.6rem 0.85rem',
                          marginBottom: '0.75rem',
                          fontSize: '0.85rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-purple)', fontWeight: 600, marginBottom: '0.3rem' }}>
                            <Brain size={15} className="pulse-icon" /> Thinking in corso...
                          </div>
                          <div style={{ fontFamily: 'monospace', fontSize: '0.82rem', opacity: 0.9, whiteSpace: 'pre-wrap', maxHeight: '180px', overflowY: 'auto' }}>
                            {thinking}
                          </div>
                        </div>
                      ) : (
                        <div style={{ marginBottom: text ? '0.75rem' : '0' }}>
                          <button
                            onClick={() => toggleThinking(idx)}
                            style={{
                              background: 'rgba(139, 92, 246, 0.15)',
                              border: '1px solid rgba(139, 92, 246, 0.3)',
                              borderRadius: '8px',
                              padding: '0.45rem 0.8rem',
                              color: '#c084fc',
                              fontSize: '0.82rem',
                              fontWeight: 600,
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.45rem',
                              transition: 'all 0.2s ease'
                            }}
                          >
                            <Brain size={15} />
                            <span>{isExpanded ? 'Nascondi Thinking' : `Mostra Step Thinking (${thinking.length} car.)`}</span>
                            {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          </button>

                          {isExpanded && (
                            <div style={{
                              marginTop: '0.5rem',
                              background: 'rgba(15, 23, 42, 0.85)',
                              border: '1px dashed rgba(139, 92, 246, 0.4)',
                              borderRadius: '8px',
                              padding: '0.75rem 0.9rem',
                              fontFamily: 'Consolas, Monaco, monospace',
                              fontSize: '0.83rem',
                              lineHeight: '1.5',
                              color: 'var(--text-muted)',
                              whiteSpace: 'pre-wrap',
                              maxHeight: '300px',
                              overflowY: 'auto'
                            }}>
                              {thinking}
                            </div>
                          )}
                        </div>
                      )
                    )}

                    {/* Main Response / Text Content */}
                    {isUser ? (
                      <div style={{ whiteSpace: 'pre-wrap' }}>{text}</div>
                    ) : text ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          code: CodeBlock
                        }}
                      >
                        {text}
                      </ReactMarkdown>
                    ) : (
                      isCurrentlyGeneratingThis && (
                        <span className="pulse-icon">
                          {thinking ? 'Generazione risposta finale...' : 'Generazione in corso...'}
                        </span>
                      )
                    )}

                    {!isUser && msg.content && (
                      <button
                        onClick={() => copyToClipboard(msg.content, idx)}
                        style={{
                          position: 'absolute',
                          top: '0.5rem',
                          right: '0.5rem',
                          background: 'none',
                          border: 'none',
                          color: 'var(--text-muted)',
                          cursor: 'pointer',
                          opacity: 0.7
                        }}
                        title="Copia messaggio completo"
                      >
                        {copiedIndex === idx ? <Check size={14} color="var(--accent-green)" /> : <Copy size={14} />}
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Controls */}
        <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid var(--border-color)', background: 'rgba(13, 19, 33, 0.7)' }}>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={isServerRunning ? "Invia un messaggio all'NPU... (Invio per inviare, Shift+Invio per nuova riga)" : "Avvia prima il server per chattare!"}
              disabled={!isServerRunning || isGenerating}
              style={{
                flex: 1,
                padding: '0.8rem 1rem',
                borderRadius: '10px',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-color)',
                color: 'white',
                fontSize: '0.95rem',
                resize: 'none',
                height: '52px',
                outline: 'none'
              }}
            />

            {isGenerating ? (
              <button className="btn-danger" onClick={handleStop} style={{ height: '52px' }}>
                <StopCircle size={20} /> Ferma
              </button>
            ) : (
              <button className="btn-primary" onClick={() => handleSend()} disabled={!isServerRunning || !input.trim()} style={{ height: '52px' }}>
                <Send size={18} /> Invia
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Settings Drawer */}
      {showSettings && (
        <div style={{
          width: '340px',
          borderLeft: '1px solid var(--border-color)',
          background: 'rgba(19, 27, 46, 0.85)',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          overflowY: 'auto'
        }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sliders size={18} color="var(--accent-purple)" /> Opzioni & Parametri
          </h3>

          {/* Thinking Toggle */}
          <div style={{
            background: 'rgba(139, 92, 246, 0.1)',
            border: '1px solid rgba(139, 92, 246, 0.25)',
            borderRadius: '10px',
            padding: '0.85rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Brain size={16} color="var(--accent-purple)" /> Thinking (CoT)
              </span>
              <button
                onClick={() => setEnableThinking(!enableThinking)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: enableThinking ? 'var(--accent-purple)' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                {enableThinking ? <ToggleRight size={28} /> : <ToggleLeft size={28} />}
              </button>
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              {enableThinking ? 'Abilitato: mostra i passaggi di ragionamento del modello.' : 'Disabilitato: il modello risponde direttamente senza thinking.'}
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              Prompt di Sistema
            </label>
            <textarea
              value={systemPrompt}
              onChange={e => {
                const val = e.target.value;
                setSystemPrompt(val);
                if (activeConvId) {
                  const conv = getConversation(activeConvId);
                  if (conv) {
                    conv.systemPrompt = val;
                    saveConversation(conv);
                  }
                }
              }}
              rows={3}
              style={{
                width: '100%',
                padding: '0.6rem',
                borderRadius: '8px',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-color)',
                color: 'white',
                fontSize: '0.85rem'
              }}
            />
          </div>

          {/* Contesto Massimo */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>Contesto Massimo (Token)</span>
              <input
                type="number"
                min="512"
                max="131072"
                step="512"
                value={maxContextLength}
                onChange={e => setMaxContextLength(Math.max(512, Math.min(131072, parseInt(e.target.value) || 4096)))}
                style={{
                  width: '80px',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '6px',
                  background: 'rgba(0,0,0,0.4)',
                  border: '1px solid var(--border-color)',
                  color: 'white',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  textAlign: 'right'
                }}
              />
            </div>
            <input
              type="range"
              min="512"
              max="131072"
              step="512"
              value={maxContextLength}
              onChange={e => setMaxContextLength(parseInt(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }}
            />
            <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
              {[4096, 8192, 16384, 32768, 131072].map(val => (
                <button
                  key={val}
                  onClick={() => setMaxContextLength(val)}
                  style={{
                    padding: '0.2rem 0.5rem',
                    fontSize: '0.72rem',
                    borderRadius: '4px',
                    border: '1px solid var(--border-color)',
                    background: maxContextLength === val ? 'var(--accent-purple)' : 'rgba(0,0,0,0.3)',
                    color: maxContextLength === val ? 'white' : 'var(--text-muted)',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  {val >= 1024 ? `${val / 1024}K` : val}
                </button>
              ))}
            </div>
          </div>

          {/* Max Tokens Risposta */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>Max Tokens Risposta</span>
              <input
                type="number"
                min="16"
                max="32768"
                step="64"
                value={maxTokens}
                onChange={e => setMaxTokens(Math.max(16, Math.min(32768, parseInt(e.target.value) || 512)))}
                style={{
                  width: '80px',
                  padding: '0.2rem 0.4rem',
                  borderRadius: '6px',
                  background: 'rgba(0,0,0,0.4)',
                  border: '1px solid var(--border-color)',
                  color: 'white',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  textAlign: 'right'
                }}
              />
            </div>
            <input
              type="range"
              min="16"
              max="16384"
              step="64"
              value={maxTokens}
              onChange={e => setMaxTokens(parseInt(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }}
            />
            <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
              {[512, 1024, 2048, 4096, 8192, 16384].map(val => (
                <button
                  key={val}
                  onClick={() => setMaxTokens(val)}
                  style={{
                    padding: '0.2rem 0.5rem',
                    fontSize: '0.72rem',
                    borderRadius: '4px',
                    border: '1px solid var(--border-color)',
                    background: maxTokens === val ? 'var(--accent-purple)' : 'rgba(0,0,0,0.3)',
                    color: maxTokens === val ? 'white' : 'var(--text-muted)',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  {val >= 1024 ? `${val / 1024}K` : val}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>Temperature</span>
              <span style={{ color: 'white', fontWeight: 600 }}>{temperature}</span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={e => setTemperature(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>Top-P</span>
              <span style={{ color: 'white', fontWeight: 600 }}>{topP}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={topP}
              onChange={e => setTopP(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
