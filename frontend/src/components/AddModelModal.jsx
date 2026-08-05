import React, { useState, useEffect, useRef } from 'react';
import { X, Download, HardDrive, Cpu, Terminal, CheckCircle2, AlertCircle, Sparkles, FolderOpen, Code, Layers, Info } from 'lucide-react';

export default function AddModelModal({ apiBase, isOpen, onClose, onSetupComplete }) {
  const [mode, setMode] = useState('auto'); // 'auto' | 'manual'
  const [supportedModels, setSupportedModels] = useState([]);
  
  // Auto Mode State
  const [selectedSupported, setSelectedSupported] = useState('gemma4');
  const [autoAlias, setAutoAlias] = useState('gemma4');
  const [autoUrl, setAutoUrl] = useState('');
  const [autoFilename, setAutoFilename] = useState('');

  // Manual Mode State
  const [manualAlias, setManualAlias] = useState('custom-model');
  const [localGgufPath, setLocalGgufPath] = useState('');
  const [manualArch, setManualArch] = useState('gemma4');
  const [customScriptPath, setCustomScriptPath] = useState('');

  // Progress / Setup Task State
  const [isSettingUp, setIsSettingUp] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');

  const logEndRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      fetchSupportedModels();
      checkOngoingSetup();
    }
  }, [isOpen]);

  useEffect(() => {
    let pollInterval = null;
    if (isSettingUp) {
      pollInterval = setInterval(fetchSetupStatus, 1500);
    }
    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [isSettingUp]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const fetchSupportedModels = async () => {
    try {
      const res = await fetch(`${apiBase}/api/supported_models`);
      const data = await res.json();
      setSupportedModels(data);
      if (data && data.length > 0) {
        const first = data[0];
        setSelectedSupported(first.id);
        setAutoAlias(first.id);
        setAutoUrl(first.default_url);
        setAutoFilename(first.filename || '');
      }
    } catch (e) {
      console.error("Failed to fetch supported models:", e);
    }
  };

  const handleSelectSupported = (modelId) => {
    const found = supportedModels.find(m => m.id === modelId);
    if (found) {
      setSelectedSupported(found.id);
      setAutoAlias(found.id);
      setAutoUrl(found.default_url);
      setAutoFilename(found.filename || '');
    }
  };

  const checkOngoingSetup = async () => {
    try {
      const res = await fetch(`${apiBase}/api/models/setup/status`);
      const data = await res.json();
      if (data.is_running) {
        setIsSettingUp(true);
        setProgress(data.progress || 0);
        setCurrentStep(data.step || '');
        setLogs(data.logs || []);
      }
    } catch (e) {
      console.error("Failed to check setup status:", e);
    }
  };

  const fetchSetupStatus = async () => {
    try {
      const res = await fetch(`${apiBase}/api/models/setup/status`);
      const data = await res.json();
      setProgress(data.progress || 0);
      setCurrentStep(data.step || '');
      setLogs(data.logs || []);
      
      if (!data.is_running) {
        setIsSettingUp(false);
        if (data.step === 'complete') {
          setProgress(100);
          if (onSetupComplete) onSetupComplete(data.active_alias);
        } else if (data.error) {
          setErrorMessage(data.error);
        }
      }
    } catch (e) {
      console.error("Error polling setup status:", e);
    }
  };

  const handleStartSetup = async () => {
    setErrorMessage('');
    setIsSettingUp(true);
    setProgress(5);
    setLogs([]);

    const payload = mode === 'auto'
      ? {
          alias: autoAlias.trim() || selectedSupported,
          arch: supportedModels.find(m => m.id === selectedSupported)?.arch || 'gemma4',
          source_type: 'auto',
          url_or_repo: autoUrl.trim(),
          filename: autoFilename.trim() || undefined
        }
      : {
          alias: manualAlias.trim() || 'custom-model',
          arch: manualArch,
          source_type: 'local',
          local_gguf_path: localGgufPath.trim(),
          custom_script: manualArch === 'custom' ? customScriptPath.trim() : undefined
        };

    try {
      const res = await fetch(`${apiBase}/api/models/setup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Impossibile avviare il setup del modello");
      }
    } catch (e) {
      setIsSettingUp(false);
      setErrorMessage(e.message);
    }
  };

  if (!isOpen) return null;

  const currentSupportedModel = supportedModels.find(m => m.id === selectedSupported);

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(5, 8, 18, 0.85)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '1.5rem'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '720px',
        maxHeight: '90vh',
        background: 'var(--bg-card)',
        border: '1px solid var(--border-color)',
        borderRadius: '16px',
        boxShadow: '0 20px 50px rgba(0, 0, 0, 0.6)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>

        {/* Modal Header */}
        <div style={{
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(255, 255, 255, 0.02)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              background: 'var(--gradient-brand)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Download size={20} color="white" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, color: 'white' }}>
                Aggiungi / Installa Modello NPU
              </h2>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Download automatico 1-click o importazione manuale con quantizzatore custom
              </div>
            </div>
          </div>
          
          <button
            onClick={onClose}
            disabled={isSettingUp}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: isSettingUp ? 'not-allowed' : 'pointer',
              padding: '0.4rem',
              borderRadius: '8px'
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Content */}
        <div style={{ padding: '1.5rem', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          
          {/* Mode Switcher */}
          {!isSettingUp && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '0.5rem',
              background: 'rgba(0,0,0,0.3)',
              padding: '0.3rem',
              borderRadius: '10px',
              border: '1px solid var(--border-color)'
            }}>
              <button
                onClick={() => setMode('auto')}
                style={{
                  padding: '0.65rem',
                  borderRadius: '8px',
                  border: 'none',
                  background: mode === 'auto' ? 'var(--accent-purple)' : 'transparent',
                  color: mode === 'auto' ? 'white' : 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '0.88rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.2s'
                }}
              >
                <Sparkles size={16} />
                <span>Automatico (1-Click)</span>
              </button>

              <button
                onClick={() => setMode('manual')}
                style={{
                  padding: '0.65rem',
                  borderRadius: '8px',
                  border: 'none',
                  background: mode === 'manual' ? 'var(--accent-purple)' : 'transparent',
                  color: mode === 'manual' ? 'white' : 'var(--text-muted)',
                  fontWeight: 600,
                  fontSize: '0.88rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.2s'
                }}
              >
                <HardDrive size={16} />
                <span>Manuale / Custom</span>
              </button>
            </div>
          )}

          {/* Mode 1: Automatic Mode */}
          {mode === 'auto' && !isSettingUp && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                  Seleziona Modello Supportato
                </label>
                <select
                  value={selectedSupported}
                  onChange={(e) => handleSelectSupported(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.7rem 0.9rem',
                    borderRadius: '8px',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontSize: '0.9rem',
                    fontWeight: 600
                  }}
                >
                  {supportedModels.map(m => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.size_approx})
                    </option>
                  ))}
                </select>
              </div>

              {currentSupportedModel && (
                <div style={{
                  padding: '0.9rem 1.1rem',
                  borderRadius: '10px',
                  background: 'rgba(139, 92, 246, 0.08)',
                  border: '1px solid rgba(139, 92, 246, 0.25)',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem'
                }}>
                  <Info size={20} color="var(--accent-purple)" style={{ marginTop: '2px', flexShrink: 0 }} />
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'white', marginBottom: '0.2rem' }}>
                      {currentSupportedModel.name}
                    </div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                      {currentSupportedModel.description}
                    </div>
                  </div>
                </div>
              )}

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                  Link HuggingFace / Repository GGUF (Modificabile)
                </label>
                <input
                  type="text"
                  value={autoUrl}
                  onChange={(e) => setAutoUrl(e.target.value)}
                  placeholder="https://huggingface.co/... o repo/id"
                  style={{
                    width: '100%',
                    padding: '0.65rem 0.85rem',
                    borderRadius: '8px',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontSize: '0.85rem'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                  Nome Identificativo (Alias)
                </label>
                <input
                  type="text"
                  value={autoAlias}
                  onChange={(e) => setAutoAlias(e.target.value)}
                  placeholder="es. gemma4, gemma4-e4b"
                  style={{
                    width: '100%',
                    padding: '0.65rem 0.85rem',
                    borderRadius: '8px',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontSize: '0.85rem'
                  }}
                />
              </div>
            </div>
          )}

          {/* Mode 2: Manual Mode */}
          {mode === 'manual' && !isSettingUp && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                  Percorso File GGUF Locale
                </label>
                <input
                  type="text"
                  value={localGgufPath}
                  onChange={(e) => setLocalGgufPath(e.target.value)}
                  placeholder="/home/user/downloads/model.gguf"
                  style={{
                    width: '100%',
                    padding: '0.65rem 0.85rem',
                    borderRadius: '8px',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontSize: '0.85rem'
                  }}
                />
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                  Inserisci il percorso assoluto del file .gguf già scaricato sul tuo sistema.
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                    Architettura / Algoritmo
                  </label>
                  <select
                    value={manualArch}
                    onChange={(e) => setManualArch(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.65rem 0.85rem',
                      borderRadius: '8px',
                      background: 'rgba(0,0,0,0.4)',
                      border: '1px solid var(--border-color)',
                      color: 'white',
                      fontSize: '0.85rem'
                    }}
                  >
                    <option value="gemma4">Gemma 4 12B (gemma4)</option>
                    <option value="gemma4-e4b">Gemma 4 E4B (gemma4-e4b)</option>
                    <option value="gemma3">Gemma 3 1B (gemma3)</option>
                    <option value="llama">Llama 3.2 (llama)</option>
                    <option value="custom">★ Algoritmo Custom (Plugin Python)</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                    Nome Modello (Alias)
                  </label>
                  <input
                    type="text"
                    value={manualAlias}
                    onChange={(e) => setManualAlias(e.target.value)}
                    placeholder="my-custom-model"
                    style={{
                      width: '100%',
                      padding: '0.65rem 0.85rem',
                      borderRadius: '8px',
                      background: 'rgba(0,0,0,0.4)',
                      border: '1px solid var(--border-color)',
                      color: 'white',
                      fontSize: '0.85rem'
                    }}
                  />
                </div>
              </div>

              {manualArch === 'custom' && (
                <div style={{
                  padding: '1rem',
                  borderRadius: '10px',
                  background: 'rgba(15, 23, 42, 0.9)',
                  border: '1px dashed var(--accent-purple)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, color: 'white', marginBottom: '0.4rem', fontSize: '0.85rem' }}>
                    <Code size={16} color="var(--accent-purple)" />
                    <span>Percorso Script Python Quantizzatore Custom</span>
                  </div>
                  <input
                    type="text"
                    value={customScriptPath}
                    onChange={(e) => setCustomScriptPath(e.target.value)}
                    placeholder="/home/user/my_quantizer_plugin.py"
                    style={{
                      width: '100%',
                      padding: '0.65rem 0.85rem',
                      borderRadius: '8px',
                      background: 'rgba(0,0,0,0.5)',
                      border: '1px solid var(--border-color)',
                      color: 'white',
                      fontSize: '0.85rem'
                    }}
                  />
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem', lineHeight: '1.4' }}>
                    Lo script deve sottoclassare <code>BaseQuantizer</code> o definire la funzione <code>quantize(gguf_path, out_dir)</code>. Per dettagli consulta <a href="file:///home/daino/progetti/alveare/docs/CUSTOM_QUANTIZERS.md" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-purple)' }}>docs/CUSTOM_QUANTIZERS.md</a>.
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Setup In Progress View */}
          {isSettingUp && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.4rem', fontWeight: 600 }}>
                  <span style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Cpu size={16} className="pulse-icon" />
                    <span>Configurazione modello in corso ({currentStep})...</span>
                  </span>
                  <span style={{ color: 'var(--accent-purple)' }}>{Math.round(progress)}%</span>
                </div>

                <div style={{ width: '100%', height: '8px', background: 'rgba(0,0,0,0.4)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${progress}%`,
                    height: '100%',
                    background: 'var(--gradient-brand)',
                    transition: 'width 0.4s ease'
                  }} />
                </div>
              </div>

              {/* Console Logs Box */}
              <div style={{
                background: 'rgba(10, 14, 26, 0.95)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                padding: '0.85rem',
                height: '240px',
                overflowY: 'auto',
                fontFamily: 'Consolas, Monaco, monospace',
                fontSize: '0.78rem',
                color: 'var(--text-muted)',
                lineHeight: '1.5'
              }}>
                {logs.length === 0 ? (
                  <div style={{ color: 'rgba(255,255,255,0.4)', fontStyle: 'italic' }}>Avvio processo...</div>
                ) : (
                  logs.map((l, i) => (
                    <div key={i} style={{ color: l.step === 'error' ? 'var(--accent-red)' : 'var(--text-muted)' }}>
                      <span style={{ opacity: 0.5, marginRight: '0.5rem' }}>{l.timestamp}</span>
                      <span>{l.message}</span>
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </div>
            </div>
          )}

          {/* Error Banner */}
          {errorMessage && (
            <div style={{
              padding: '0.85rem 1rem',
              borderRadius: '8px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: '#fca5a5',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <AlertCircle size={18} />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Complete Success Banner */}
          {progress === 100 && !isSettingUp && (
            <div style={{
              padding: '0.85rem 1rem',
              borderRadius: '8px',
              background: 'rgba(34, 197, 94, 0.1)',
              border: '1px solid rgba(34, 197, 94, 0.3)',
              color: '#86efac',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <CheckCircle2 size={18} />
              <span>Modello installato e configurato con successo per la NPU!</span>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div style={{
          padding: '1rem 1.5rem',
          borderTop: '1px solid var(--border-color)',
          background: 'rgba(0, 0, 0, 0.2)',
          display: 'flex',
          justify: 'flex-end',
          gap: '0.75rem'
        }}>
          <button
            onClick={onClose}
            disabled={isSettingUp}
            style={{
              padding: '0.6rem 1.1rem',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              background: 'transparent',
              color: 'white',
              cursor: isSettingUp ? 'not-allowed' : 'pointer',
              fontSize: '0.85rem',
              fontWeight: 600
            }}
          >
            {progress === 100 ? 'Chiudi' : 'Annulla'}
          </button>

          {progress !== 100 && (
            <button
              onClick={handleStartSetup}
              disabled={isSettingUp}
              style={{
                padding: '0.6rem 1.4rem',
                borderRadius: '8px',
                border: 'none',
                background: 'var(--accent-purple)',
                color: 'white',
                cursor: isSettingUp ? 'not-allowed' : 'pointer',
                fontSize: '0.85rem',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                boxShadow: '0 0 15px rgba(139, 92, 246, 0.4)'
              }}
            >
              {isSettingUp ? (
                <>
                  <Cpu size={16} className="pulse-icon" />
                  <span>Installazione in corso...</span>
                </>
              ) : (
                <>
                  <Download size={16} />
                  <span>Installa Modello</span>
                </>
              )}
            </button>
          )}
        </div>

      </div>
    </div>
  );
}
