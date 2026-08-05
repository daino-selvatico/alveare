import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Trash2, Sliders, Cpu, Zap, StopCircle, Copy, Check, Brain, ChevronDown, ChevronUp, ToggleLeft, ToggleRight, Paperclip, Image as ImageIcon, Music, FileText, Upload, X, File, Eye } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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

function CodeBlock({ node, inline, className, children, ...props }) {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const codeString = String(children).replace(/\n$/, '');

  if (inline) {
    return (
      <code style={{
        background: 'rgba(0, 0, 0, 0.4)',
        padding: '0.15rem 0.4rem',
        borderRadius: '4px',
        fontFamily: 'monospace',
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
        <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--accent-purple)' }}>
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
        fontFamily: 'Consolas, Monaco, monospace',
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
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Ciao! Sono il modello LLM in esecuzione sull\'NPU AMD Ryzen AI. Come posso aiutarti oggi?' }
  ]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [expandedThinking, setExpandedThinking] = useState({});

  const [attachments, setAttachments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [acceptFilter, setAcceptFilter] = useState('*/*');
  const [isDragging, setIsDragging] = useState(false);
  const [previewImageModal, setPreviewImageModal] = useState(null);
  const [expandedDocs, setExpandedDocs] = useState({});

  const fileInputRef = useRef(null);
  const uploadMenuRef = useRef(null);
  
  const [systemPrompt, setSystemPrompt] = useState('Sei un assistente AI esperto ed utile.');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(512);
  const [maxContextLength, setMaxContextLength] = useState(4096);
  const [enableThinking, setEnableThinking] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  const [metrics, setMetrics] = useState({ tokens: 0, elapsed: 0, tps: 0 });

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

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

  const toggleDocExpand = (docId) => {
    setExpandedDocs(prev => ({ ...prev, [docId]: !prev[docId] }));
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

    setIsUploading(true);
    const readPromises = Array.from(fileList).map(file => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result || '';
          const b64Data = typeof result === 'string' && result.includes(',') ? result.split(',')[1] : result;
          resolve({
            filename: file.name,
            content_b64: b64Data,
            mime_type: file.type || undefined
          });
        };
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
      });
    });

    try {
      const payloadFiles = await Promise.all(readPromises);
      const res = await fetch(`${apiBase}/api/upload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: payloadFiles })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Errore caricamento file (${res.status})`);
      }

      const data = await res.json();
      if (data.files && data.files.length > 0) {
        setAttachments(prev => [...prev, ...data.files]);
      }
    } catch (err) {
      alert(`Impossibile caricare i file: ${err.message}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemoveAttachment = (file_id) => {
    setAttachments(prev => prev.filter(a => a.file_id !== file_id));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
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

  const handleSend = async () => {
    if ((!input.trim() && attachments.length === 0) || isGenerating) return;
    if (!isServerRunning) {
      alert("Il server di inferenza non è in esecuzione! Avvialo dal Control Panel.");
      return;
    }

    let promptContent = input.trim();
    if (attachments.length > 0) {
      const fileContexts = attachments.map(att => att.extracted_text).join('\n\n');
      promptContent = promptContent ? `${promptContent}\n\n${fileContexts}` : fileContexts;
      promptContent = promptContent.trim();
    }

    const userMsg = {
      role: 'user',
      content: promptContent,
      displayText: input.trim() || undefined,
      attachments: [...attachments]
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setAttachments([]);
    setIsGenerating(true);

    const assistantIndex = newMessages.length;
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    const chatMessages = [];
    if (systemPrompt.trim()) {
      chatMessages.push({ role: 'system', content: systemPrompt.trim() });
    }

    for (const msg of newMessages) {
      chatMessages.push({ role: msg.role, content: msg.content });
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const startTime = Date.now();
    let tokenCount = 0;

    try {
      const response = await fetch(`${apiBase}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel,
          messages: chatMessages,
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
          if (trimmed === 'data: [DONE]') break;
          if (trimmed.startsWith('data: ')) {
            const jsonStr = trimmed.slice(6);
            try {
              const data = JSON.parse(jsonStr);
              const delta = data.choices?.[0]?.delta?.content || '';
              if (delta) {
                tokenCount++;
                setMessages(prev => {
                  const updated = [...prev];
                  if (updated[assistantIndex]) {
                    updated[assistantIndex] = {
                      ...updated[assistantIndex],
                      content: updated[assistantIndex].content + delta
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
            } catch (e) {}
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages(prev => {
          const updated = [...prev];
          if (updated[assistantIndex]) {
            updated[assistantIndex] = {
              ...updated[assistantIndex],
              content: updated[assistantIndex].content + `\n\n⚠️ [Errore]: ${err.message}`
            };
          }
          return updated;
        });
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    setIsGenerating(false);
  };

  const handleClear = () => {
    setMessages([{ role: 'assistant', content: 'Conversazione azzerata. Come posso aiutarti?' }]);
    setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
    setExpandedThinking({});
    setAttachments([]);
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} style={{ display: 'flex', height: 'calc(100vh - 70px)', width: '100%', position: 'relative' }}>
      
      {isDragging && (
        <div style={{ position: 'absolute', inset: 0, background: 'rgba(15, 23, 42, 0.88)', backdropFilter: 'blur(8px)', zIndex: 500, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '3px dashed var(--accent-purple)', margin: '0.75rem', borderRadius: '16px', pointerEvents: 'none' }}>
          <Upload size={54} color="var(--accent-purple)" className="pulse-icon" />
          <h3 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'white', marginTop: '1rem' }}>Rilascia i file qui per caricarli</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.3rem' }}>Supporta immagini, audio, documenti (PDF, TXT, MD, CSV, Codice) e file generici</p>
        </div>
      )}

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
        <div style={{ padding: '0.85rem 1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(19, 27, 46, 0.5)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontWeight: 600, fontSize: '1rem', color: 'white' }}>Playground Chat Multimodale</span>
            <span className="badge badge-success" style={{ textTransform: 'none' }}><Cpu size={14} /> Modello attivo: {activeModel}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {metrics.tps > 0 && <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', border: '1px solid rgba(6, 182, 212, 0.3)' }}><Zap size={14} /> {metrics.tps} tok/s ({metrics.tokens} token in {metrics.elapsed}s)</span>}
            <button className="btn-secondary" onClick={handleClear} title="Reset"><Trash2 size={16} /> Reset</button>
            <button className="btn-secondary" onClick={() => setShowSettings(!showSettings)}><Sliders size={16} /> Impostazioni</button>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {messages.map((msg, idx) => {
            const isUser = msg.role === 'user';
            const { thinking, text } = parseThinking(msg.content);
            const displayMessageText = msg.displayText !== undefined ? msg.displayText : text;
            const isCurrentlyGeneratingThis = isGenerating && idx === messages.length - 1;
            const isExpanded = !!expandedThinking[idx];

            return (
              <div key={idx} style={{ display: 'flex', gap: '1rem', maxWidth: '85%', alignSelf: isUser ? 'flex-end' : 'flex-start', flexDirection: isUser ? 'row-reverse' : 'row' }}>
                <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: isUser ? 'var(--gradient-brand)' : 'rgba(139, 92, 246, 0.2)', border: !isUser ? '1px solid var(--accent-purple)' : 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  {isUser ? <User size={18} color="white" /> : <Bot size={18} color="#c084fc" />}
                </div>
                <div style={{ background: isUser ? 'rgba(139, 92, 246, 0.18)' : 'rgba(30, 41, 59, 0.7)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '0.85rem 1.1rem', color: 'var(--text-main)', fontSize: '0.95rem', lineHeight: '1.6', position: 'relative', minWidth: '240px' }}>
                  {msg.attachments && msg.attachments.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.65rem', marginBottom: '0.75rem' }}>
                      {msg.attachments.map((att, aIdx) => {
                        const docId = `msg-${idx}-att-${aIdx}`;
                        return (
                          <div key={aIdx} style={{ background: 'rgba(15, 23, 42, 0.85)', border: '1px solid rgba(139, 92, 246, 0.3)', borderRadius: '10px', padding: '0.5rem 0.75rem', display: 'flex', flexDirection: 'column', gap: '0.4rem', maxWidth: '320px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              {att.file_type === 'image' && <ImageIcon size={16} color="#f472b6" />}
                              {att.file_type === 'audio' && <Music size={16} color="#06b6d4" />}
                              {att.file_type === 'document' && <FileText size={16} color="#a78bfa" />}
                              <span style={{ fontSize: '0.83rem', fontWeight: 600, color: 'white' }}>{att.filename}</span>
                              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>{att.size_formatted}</span>
                            </div>
                            {att.file_type === 'image' && att.preview_url && <img src={att.preview_url} style={{ width: '100%', height: 'auto', borderRadius: '6px', cursor: 'pointer' }} onClick={() => setPreviewImageModal(att.preview_url)} />}
                            {att.file_type === 'audio' && att.preview_url && <audio controls src={att.preview_url} style={{ width: '100%', height: '32px' }} />}
                            {att.file_type === 'document' && att.extracted_text && (
                              <div>
                                <button onClick={() => toggleDocExpand(docId)} style={{ background: 'none', border: 'none', color: '#c084fc', fontSize: '0.74rem', cursor: 'pointer' }}>{expandedDocs[docId] ? 'Nascondi' : 'Mostra'} Testo</button>
                                {expandedDocs[docId] && <pre style={{ fontSize: '0.75rem', overflow: 'auto', maxHeight: '100px', whiteSpace: 'pre-wrap' }}>{att.extracted_text}</pre>}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {!isUser && thinking && (isCurrentlyGeneratingThis ? <div>Thinking...</div> : <div><button onClick={() => toggleThinking(idx)} style={{ color: '#c084fc' }}>{isExpanded ? 'Nascondi' : 'Mostra'} Thinking</button>{isExpanded && <pre>{thinking}</pre>}</div>)}
                  {isUser ? <div style={{ whiteSpace: 'pre-wrap' }}>{displayMessageText}</div> : <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ code: CodeBlock }}>{text}</ReactMarkdown>}
                  {!isUser && msg.content && <button onClick={() => copyToClipboard(msg.content, idx)} style={{ position: 'absolute', top: '0.5rem', right: '0.5rem', background: 'none', border: 'none' }}>{copiedIndex === idx ? <Check size={14} color="var(--accent-green)" /> : <Copy size={14} />}</button>}
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid var(--border-color)', background: 'rgba(13, 19, 33, 0.7)', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {attachments.length > 0 && <div style={{ display: 'flex', gap: '0.5rem' }}>{attachments.map(att => <div key={att.file_id} style={{ background: 'rgba(30, 41, 59, 0.9)', padding: '0.35rem', borderRadius: '8px' }}>{att.filename} <button onClick={() => handleRemoveAttachment(att.file_id)}><X size={12} /></button></div>)}</div>}
          <input type="file" ref={fileInputRef} accept={acceptFilter} multiple onChange={e => handleFilesSelected(e.target.files)} style={{ display: 'none' }} />
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <div style={{ position: 'relative' }} ref={uploadMenuRef}>
              <button className="btn-secondary" onClick={() => setShowUploadMenu(!showUploadMenu)} disabled={!isServerRunning || isGenerating}><Paperclip size={18} /></button>
              {showUploadMenu && <div style={{ position: 'absolute', bottom: '60px', background: '#131b2e', padding: '0.5rem', borderRadius: '12px' }}>
                <button onClick={() => handleOpenUpload('image')}>Immagine</button>
                <button onClick={() => handleOpenUpload('audio')}>Audio</button>
                <button onClick={() => handleOpenUpload('document')}>Documento</button>
              </div>}
            </div>
            <textarea value={input} onChange={e => setInput(e.target.value)} disabled={!isServerRunning || isGenerating} style={{ flex: 1, padding: '0.8rem', borderRadius: '10px' }} />
            {isGenerating ? <button onClick={handleStop}><StopCircle size={20} /></button> : <button onClick={handleSend} disabled={!isServerRunning || (!input.trim() && attachments.length === 0)}><Send size={18} /></button>}
          </div>
        </div>
      </div>
      {previewImageModal && <div onClick={() => setPreviewImageModal(null)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><img src={previewImageModal} style={{ maxWidth: '90%' }} /></div>}
      {showSettings && <div style={{ width: '340px', padding: '1.25rem' }}>Settings...</div>}
    </div>
  );
}
