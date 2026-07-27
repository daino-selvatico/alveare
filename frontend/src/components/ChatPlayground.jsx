import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Trash2, Sliders, RefreshCw, Cpu, Zap, StopCircle, Copy, Check } from 'lucide-react';

export default function ChatPlayground({ apiBase, activeModel, isServerRunning }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Ciao! Sono il modello LLM in esecuzione sull\'NPU AMD Ryzen AI. Come posso aiutarti oggi?' }
  ]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  
  // Generation parameters
  const [systemPrompt, setSystemPrompt] = useState('Sei un assistente AI esperto ed utile.');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(512);
  const [showSettings, setShowSettings] = useState(false);

  // Performance metrics
  const [metrics, setMetrics] = useState({ tokens: 0, elapsed: 0, tps: 0 });

  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = () => {
    if (!input.trim() || isGenerating) return;
    if (!isServerRunning) {
      alert("Il server di inferenza non è in esecuzione! Avvialo dal Control Panel.");
      return;
    }

    const userMsg = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setIsGenerating(true);

    // Placeholder assistant message
    const assistantIndex = newMessages.length;
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    // Establish WebSocket connection
    const wsUrl = apiBase.replace(/^http/, 'ws') + '/ws/chat';
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      // Build request payload
      const chatMessages = [];
      if (systemPrompt.trim()) {
        chatMessages.push({ role: 'system', content: systemPrompt.trim() });
      }
      chatMessages.push(...newMessages);

      ws.send(JSON.stringify({
        action: 'chat',
        messages: chatMessages,
        model: activeModel,
        temperature,
        top_p: topP,
        top_k: topK,
        max_tokens: maxTokens
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          setMessages(prev => {
            const updated = [...prev];
            if (updated[assistantIndex]) {
              updated[assistantIndex] = {
                ...updated[assistantIndex],
                content: updated[assistantIndex].content + data.content
              };
            }
            return updated;
          });
        } else if (data.type === 'done') {
          setIsGenerating(false);
          if (data.metrics) {
            setMetrics({
              tokens: data.metrics.tokens || 0,
              elapsed: data.metrics.elapsed_seconds || 0,
              tps: data.metrics.tps || 0
            });
          }
          ws.close();
        } else if (data.type === 'error') {
          setIsGenerating(false);
          setMessages(prev => {
            const updated = [...prev];
            if (updated[assistantIndex]) {
              updated[assistantIndex].content += `\n\n⚠️ [Errore]: ${data.message}`;
            }
            return updated;
          });
          ws.close();
        }
      } catch (err) {
        console.error("WS Parse error:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WS Error:", err);
      setIsGenerating(false);
      ws.close();
    };

    ws.onclose = () => {
      setIsGenerating(false);
    };
  };

  const handleStop = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsGenerating(false);
  };

  const handleClear = () => {
    setMessages([
      { role: 'assistant', content: 'Conversazione azzerata. Come posso aiutarti?' }
    ]);
    setMetrics({ tokens: 0, elapsed: 0, tps: 0 });
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 70px)', width: '100%' }}>
      
      {/* Main Chat Workspace */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
        
        {/* Top Chat Bar */}
        <div style={{
          padding: '0.85rem 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(19, 27, 46, 0.5)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontWeight: 600, fontSize: '1rem', color: 'white' }}>
              Playground Realtime WebSocket
            </span>
            <span className="badge badge-success" style={{ textTransform: 'none' }}>
              <Cpu size={14} /> Modello attivo: {activeModel}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {metrics.tps > 0 && (
              <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
                <Zap size={14} /> {metrics.tps} tok/s ({metrics.tokens} token in {metrics.elapsed}s)
              </span>
            )}
            
            <button className="btn-secondary" onClick={handleClear} title="Cancella conversazione">
              <Trash2 size={16} /> Reset
            </button>

            <button className="btn-secondary" onClick={() => setShowSettings(!showSettings)}>
              <Sliders size={16} /> Impostazioni
            </button>
          </div>
        </div>

        {/* Message Stream */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {messages.map((msg, idx) => (
            <div key={idx} style={{
              display: 'flex',
              gap: '1rem',
              maxWidth: '85%',
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
            }}>
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: msg.role === 'user' ? 'var(--gradient-brand)' : 'rgba(139, 92, 246, 0.2)',
                border: msg.role === 'assistant' ? '1px solid var(--accent-purple)' : 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0
              }}>
                {msg.role === 'user' ? <User size={18} color="white" /> : <Bot size={18} color="#c084fc" />}
              </div>

              <div style={{
                background: msg.role === 'user' ? 'rgba(139, 92, 246, 0.18)' : 'rgba(30, 41, 59, 0.7)',
                border: '1px solid var(--border-color)',
                borderRadius: '12px',
                padding: '0.85rem 1.1rem',
                color: 'var(--text-main)',
                fontSize: '0.95rem',
                lineHeight: '1.6',
                whiteSpace: 'pre-wrap',
                position: 'relative'
              }}>
                {msg.content || (isGenerating && idx === messages.length - 1 ? <span className="pulse-icon">Generazione in corso...</span> : '')}

                {msg.role === 'assistant' && msg.content && (
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
                    title="Copia"
                  >
                    {copiedIndex === idx ? <Check size={14} color="var(--accent-green)" /> : <Copy size={14} />}
                  </button>
                )}
              </div>
            </div>
          ))}
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
              <button className="btn-primary" onClick={handleSend} disabled={!isServerRunning || !input.trim()} style={{ height: '52px' }}>
                <Send size={18} /> Invia
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Settings Drawer */}
      {showSettings && (
        <div style={{
          width: '320px',
          borderLeft: '1px solid var(--border-color)',
          background: 'rgba(19, 27, 46, 0.8)',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          overflowY: 'auto'
        }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sliders size={18} color="var(--accent-purple)" /> Parametri Sampling
          </h3>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              Prompt di Sistema
            </label>
            <textarea
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
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

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>Temperature</span>
              <span>{temperature}</span>
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
              <span>{topP}</span>
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

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              <span>Max Tokens</span>
              <span>{maxTokens}</span>
            </div>
            <input
              type="range"
              min="16"
              max="2048"
              step="16"
              value={maxTokens}
              onChange={e => setMaxTokens(parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
