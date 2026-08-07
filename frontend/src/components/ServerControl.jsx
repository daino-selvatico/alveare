import React, { useState, useEffect, useCallback } from 'react';
import { Play, Square, RotateCw, Server, HardDrive, Settings, AlertCircle, CheckCircle, Hammer, ChevronDown, ChevronUp, Plus, Trash2 } from 'lucide-react';
import AddModelModal from './AddModelModal';

export default function ServerControl({ apiBase, status, models = [], modelsLoading = false, onRefresh }) {
  const [loading, setLoading] = useState(false);
  const [targetModel, setTargetModel] = useState(status?.model || (models.length > 0 ? models[0].id : 'gemma4'));
  const [host, setHost] = useState(status?.host || '127.0.0.1');
  const [port, setPort] = useState(status?.port || 8000);
  const [legacy, setLegacy] = useState(status?.legacy || false);
  const [offline, setOffline] = useState(status?.offline || false);

  // Kernel Build States
  const [kernelStatus, setKernelStatus] = useState(null);
  const [showBuildOptions, setShowBuildOptions] = useState(false);
  const [noGemm, setNoGemm] = useState(false);
  const [maxBatch, setMaxBatch] = useState(16);
  const [buildingModel, setBuildingModel] = useState(null);
  const [isAddModelOpen, setIsAddModelOpen] = useState(false);

  const fetchKernelStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/kernels/status`);
      if (res.ok) {
        const data = await res.json();
        setKernelStatus(data);
      }
    } catch (e) {
      console.error("Failed to fetch kernel status:", e);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchKernelStatus();
  }, [status?.model, fetchKernelStatus]);

  useEffect(() => {
    if (models.length > 0 && !targetModel) {
      setTargetModel(models[0].id);
    }
  }, [models, targetModel]);

  const handleDeleteModel = async (modelId) => {
    if (status?.is_running && status?.model === modelId) {
      alert(`Impossibile eliminare '${modelId}' perché è attualmente in esecuzione. Arresta prima il server.`);
      return;
    }
    if (!confirm(`Sei sicuro di voler eliminare il modello '${modelId}'? I pesi quantizzati ed i kernel compilati verranno rimossi definitivamente da disco.`)) {
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/api/models/${modelId}`, { method: 'DELETE' });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Errore durante l'eliminazione del modello");
      }
      alert(`Modello '${modelId}' eliminato con successo.`);
      onRefresh();
    } catch (e) {
      alert(`Errore eliminazione modello: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleStart = async (modelToStart) => {
    const selected = modelToStart || targetModel;
    if (!selected) {
      alert("Seleziona prima un modello da avviare.");
      return;
    }
    setLoading(true);
    try {
      await fetch(`${apiBase}/api/control/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selected, host, port, legacy, offline })
      });
      setTimeout(() => {
        onRefresh();
        fetchKernelStatus();
      }, 1000);
    } catch (e) {
      alert(`Errore avvio server: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await fetch(`${apiBase}/api/control/stop`, { method: 'POST' });
      setTimeout(() => {
        onRefresh();
        fetchKernelStatus();
      }, 1000);
    } catch (e) {
      alert(`Errore arresto server: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRestart = async (modelToRestart) => {
    const selected = modelToRestart || targetModel;
    setLoading(true);
    try {
      await fetch(`${apiBase}/api/control/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selected, host, port, legacy, offline })
      });
      setTimeout(() => {
        onRefresh();
        fetchKernelStatus();
      }, 1500);
    } catch (e) {
      alert(`Errore riavvio server: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBuildKernels = async (modelId) => {
    if (!confirm(`Vuoi avviare la compilazione dei kernel NPU per '${modelId}'? La procedura compila i file .xclbin e aggiorna kernels/build/manifest.json.`)) {
      return;
    }
    setBuildingModel(modelId);
    try {
      await fetch(`${apiBase}/api/control/build-kernels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelId,
          no_gemm: noGemm,
          max_batch: maxBatch
        })
      });
      alert(`Compilazione kernel avviata per '${modelId}'. Puoi seguire i log in tempo reale nella tab 'Log & Diagnostica'.`);
      setTimeout(fetchKernelStatus, 3000);
    } catch (e) {
      alert(`Errore avvio compilazione kernel: ${e}`);
    } finally {
      setBuildingModel(null);
    }
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* 1. HERO SERVER STATUS BANNER */}
      <div className="glass-card" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <Server size={26} color="var(--accent-purple)" />
              <h2 style={{ fontSize: '1.4rem', fontWeight: 700 }}>Inference Engine Controller</h2>
              
              {status?.status === 'running' && status?.is_loaded && <span className="badge badge-success"><CheckCircle size={14} /> In Esecuzione ({status.model})</span>}
              {(status?.status === 'starting' || (status?.is_running && !status?.is_loaded)) && (
                <span className="badge badge-warning"><RotateCw size={14} className="pulse-icon" /> Caricamento Modello ({status?.load_progress || 0}%)</span>
              )}
              {status?.status === 'stopped' && <span className="badge badge-danger"><AlertCircle size={14} /> Server Spento</span>}
              {status?.status === 'error' && <span className="badge badge-danger"><AlertCircle size={14} /> Errore Server</span>}
              {status?.status === 'building_kernels' && <span className="badge badge-warning"><Hammer size={14} className="pulse-icon" /> Compilazione Kernel NPU...</span>}
            </div>
            
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              {status?.is_running
                ? `Server attivo su http://${status.host}:${status.port} (PID: ${status.pid}, Uptime: ${status.uptime_seconds}s)`
                : models.length > 0
                ? "Seleziona un modello dalla scheda sottostante e clicca 'Carica & Avvia'."
                : "Nessun modello trovato. Clicca 'Aggiungi Modello' per iniziare."}
            </p>
          </div>

          {/* Global Action Controls */}
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            {status?.is_running ? (
              <>
                <button className="btn-secondary" onClick={() => handleRestart(targetModel)} disabled={loading} aria-label="Riavvia server NPU">
                  <RotateCw size={18} className={loading ? "pulse-icon" : ""} /> Riavvia
                </button>
                <button className="btn-danger" onClick={handleStop} disabled={loading} aria-label="Arresta server NPU">
                  <Square size={18} /> Arresta
                </button>
              </>
            ) : (
              <button
                className="btn-primary"
                onClick={() => handleStart(targetModel)}
                disabled={loading || models.length === 0}
                style={{ background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}
                aria-label="Avvia server NPU"
              >
                <Play size={18} className={loading ? "pulse-icon" : ""} />
                {loading ? 'Avvio in corso...' : `Avvia Server NPU ${targetModel ? `(${targetModel})` : ''}`}
              </button>
            )}
          </div>
        </div>

        {/* Model Load Progress Bar */}
        {(status?.status === 'starting' || (status?.is_running && !status?.is_loaded)) && (
          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '10px',
            padding: '1rem',
            border: '1px solid rgba(245, 158, 11, 0.3)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: 'white', marginBottom: '0.4rem' }}>
              <span>{status?.load_step || "Caricamento pesi layer e inizializzazione NPU in corso..."}</span>
              <span style={{ color: '#fbbf24' }}>{status?.load_progress || 0}%</span>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${status?.load_progress || 0}%`,
                background: 'linear-gradient(90deg, #f59e0b 0%, #10b981 100%)',
                borderRadius: '4px',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        )}

        {/* Error Alert Box */}
        {(status?.status === 'error' || status?.last_error) && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            borderRadius: '10px',
            padding: '1rem 1.25rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.85rem'
          }}>
            <AlertCircle size={22} color="#ef4444" style={{ marginTop: '0.1rem', flexShrink: 0 }} />
            <div style={{ flex: 1 }}>
              <div style={{ color: '#ef4444', fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.25rem' }}>
                Errore durante l'esecuzione del Server di Inferenza
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap', maxHeight: '120px', overflowY: 'auto' }}>
                {status?.last_error || "Il server si è arrestato inaspettatamente."}
              </div>
            </div>
            <button
              className="btn-secondary"
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem', borderColor: 'rgba(239, 68, 68, 0.4)', color: '#ef4444' }}
              onClick={() => handleStart(targetModel)}
              aria-label="Riprova ad avviare il server"
            >
              Riprova
            </button>
          </div>
        )}
      </div>

      {/* 2. UNIFIED MODEL GALLERY / EMPTY ONBOARDING STATE */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-main)' }}>
            <HardDrive size={22} color="var(--accent-cyan)" /> Modelli Quantizzati Disponibili
          </h3>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <button
              onClick={() => setIsAddModelOpen(true)}
              style={{
                padding: '0.55rem 1.1rem',
                borderRadius: '8px',
                border: 'none',
                background: 'var(--accent-purple)',
                color: 'white',
                fontWeight: 700,
                fontSize: '0.88rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                boxShadow: '0 0 15px rgba(139, 92, 246, 0.4)'
              }}
              aria-label="Apri finestra aggiungi modello"
            >
              <Plus size={16} /> Aggiungi Modello
            </button>
          </div>
        </div>

        {/* Loading Skeletons */}
        {modelsLoading ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
            {[1, 2].map(n => (
              <div
                key={n}
                className="glass-card pulse-icon"
                style={{ height: '180px', borderRadius: '12px', opacity: 0.6 }}
              />
            ))}
          </div>
        ) : models.length === 0 ? (
          /* First-run Empty State Onboarding Card */
          <div
            className="glass-card"
            style={{
              padding: '3rem 2rem',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '1.25rem',
              maxWidth: '640px',
              margin: '0 auto',
              borderColor: 'rgba(139, 92, 246, 0.3)'
            }}
          >
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '16px',
              background: 'rgba(139, 92, 246, 0.15)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: 'var(--shadow-glow)'
            }}>
              <HardDrive size={32} color="var(--accent-purple)" />
            </div>

            <div>
              <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.5rem' }}>
                Nessun Modello LLM Trovato
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: '1.6' }}>
                Per poter eseguire l'inferenza sull'hardware NPU AMD Ryzen AI, è necessario installare almeno un modello quantizzato.
                Puoi scaricare con 1-click un modello supportato (come Gemma 4 12B) oppure importare un file <code>.gguf</code> locale.
              </p>
            </div>

            <button
              onClick={() => setIsAddModelOpen(true)}
              className="btn-primary"
              style={{ padding: '0.75rem 1.5rem', fontSize: '0.95rem' }}
              aria-label="Guida aggiungi modello"
            >
              <Plus size={18} /> Scarica o Importa il tuo Primo Modello
            </button>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
            {models.map(m => {
              const isServed = status?.is_running && status?.model === m.id;
              const isSelected = targetModel === m.id;
              const mKernel = kernelStatus?.[m.id] || (kernelStatus?.manifest_model_type === m.arch ? kernelStatus : null);
              const isKernelMatching = mKernel?.manifest_exists;

              return (
                <div
                  key={m.id}
                  onClick={() => setTargetModel(m.id)}
                  className="glass-card"
                  style={{
                    padding: '1.35rem',
                    cursor: 'pointer',
                    border: isServed ? '2px solid #10b981' : isSelected ? '2px solid var(--accent-purple)' : '1px solid var(--border-color)',
                    background: isServed ? 'rgba(16, 185, 129, 0.08)' : isSelected ? 'rgba(139, 92, 246, 0.1)' : 'rgba(23, 32, 54, 0.6)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '1rem'
                  }}
                >
                  {/* Header Card */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <h4 style={{ fontSize: '1.15rem', fontWeight: 700, color: isSelected ? '#c084fc' : 'var(--text-main)' }}>
                        {m.id}
                      </h4>
                      <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                        Architettura: <strong>{m.arch}</strong> • Dimensione: {m.size_mb} MB
                      </span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.3rem' }}>
                      {isServed && status?.is_loaded && <span className="badge badge-success"><CheckCircle size={12} /> IN ESECUZIONE</span>}
                      {isServed && !status?.is_loaded && <span className="badge badge-warning"><RotateCw size={12} className="pulse-icon" /> CARICAMENTO ({status?.load_progress || 0}%)</span>}
                      {isSelected && !isServed && <span className="badge" style={{ background: 'rgba(139,92,246,0.2)', color: '#c084fc', border: '1px solid rgba(139,92,246,0.4)' }}>SELEZIONATO</span>}
                    </div>
                  </div>

                  {/* Performance Speed Badge if active */}
                  {isServed && status?.tok_per_sec > 0 && (
                    <div style={{
                      fontSize: '0.83rem',
                      padding: '0.45rem 0.75rem',
                      borderRadius: '6px',
                      background: 'rgba(6, 182, 212, 0.15)',
                      border: '1px solid rgba(6, 182, 212, 0.3)',
                      color: 'var(--accent-cyan)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      fontWeight: 600
                    }}>
                      <span>Velocità Misurata:</span>
                      <span>⚡ {status.tok_per_sec} tok/s</span>
                    </div>
                  )}

                  {/* Kernel Status Indicator */}
                  <div style={{ fontSize: '0.82rem', padding: '0.5rem 0.75rem', borderRadius: '6px', background: 'rgba(0,0,0,0.35)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>Kernel Hardware NPU:</span>
                    {isKernelMatching ? (
                      <span style={{ color: '#34d399', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        ✓ Compilati ({mKernel.kernels_count} kernel)
                      </span>
                    ) : (
                      <span style={{ color: '#fbbf24', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        ⚠️ Non Compilati
                      </span>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div style={{ display: 'flex', gap: '0.6rem', marginTop: 'auto', paddingTop: '0.5rem' }}>
                    <button
                      className="btn-primary"
                      style={{ flex: 1, padding: '0.5rem', fontSize: '0.85rem', justifyContent: 'center' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setTargetModel(m.id);
                        if (isServed) {
                          handleRestart(m.id);
                        } else {
                          handleStart(m.id);
                        }
                      }}
                      disabled={loading}
                      aria-label={`Avvia o riavvia modello ${m.id}`}
                    >
                      {isServed ? <RotateCw size={14} /> : <Play size={14} />}
                      {isServed ? "Riavvia" : "Carica & Avvia"}
                    </button>

                    <button
                      className="btn-secondary"
                      style={{ padding: '0.5rem 0.75rem', fontSize: '0.85rem', color: '#c084fc', borderColor: 'rgba(139,92,246,0.4)' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setTargetModel(m.id);
                        handleBuildKernels(m.id);
                      }}
                      disabled={buildingModel === m.id}
                      title="Compila/Rigenera kernel .xclbin NPU per questo modello"
                      aria-label={`Compila kernel NPU per ${m.id}`}
                    >
                      <Hammer size={14} /> {buildingModel === m.id ? "Compilazione..." : "Compila Kernel"}
                    </button>

                    <button
                      className="btn-danger"
                      style={{ padding: '0.5rem 0.65rem', fontSize: '0.85rem' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteModel(m.id);
                      }}
                      disabled={loading || (status?.is_running && status?.model === m.id)}
                      title="Elimina modello, pesi quantizzati e kernel da disco"
                      aria-label={`Elimina modello ${m.id}`}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 3. SETTINGS & OPTIONS PANEL */}
      <div className="glass-card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Settings size={20} color="var(--accent-cyan)" /> Configurazione Server & Opzioni Avanzate Kernel
          </h3>

          <button className="btn-secondary" onClick={() => setShowBuildOptions(!showBuildOptions)} style={{ padding: '0.4rem 0.75rem', fontSize: '0.85rem' }} aria-expanded={showBuildOptions}>
            {showBuildOptions ? <ChevronUp size={16} /> : <ChevronDown size={16} />} Opzioni Avanzate Compilazione
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.25rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              Indirizzo Host
            </label>
            <input
              type="text"
              value={host}
              onChange={e => setHost(e.target.value)}
              style={{
                width: '100%',
                padding: '0.65rem',
                borderRadius: '8px',
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid var(--border-color)',
                color: 'white',
                fontFamily: 'var(--font-mono)'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              Porta API HTTP
            </label>
            <input
              type="number"
              value={port}
              onChange={e => setPort(parseInt(e.target.value) || 8000)}
              style={{
                width: '100%',
                padding: '0.65rem',
                borderRadius: '8px',
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid var(--border-color)',
                color: 'white',
                fontFamily: 'var(--font-mono)'
              }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', justifyContent: 'center' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.88rem' }}>
              <input type="checkbox" checked={legacy} onChange={e => setLegacy(e.target.checked)} />
              <span>Server Legacy Python (Uvicorn)</span>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.88rem' }}>
              <input type="checkbox" checked={offline} onChange={e => setOffline(e.target.checked)} />
              <span>Modalità Offline (HF_HUB_OFFLINE)</span>
            </label>
          </div>
        </div>

        {/* Expandable Advanced Kernel Compilation Options */}
        {showBuildOptions && (
          <div style={{ marginTop: '1.25rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.88rem' }}>
              <input type="checkbox" checked={noGemm} onChange={e => setNoGemm(e.target.checked)} />
              <span>--no-gemm (Compilazione veloce saltando forme GEMM prefill)</span>
            </label>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.88rem' }}>
              <span>--max-batch:</span>
              <input
                type="number"
                value={maxBatch}
                onChange={e => setMaxBatch(parseInt(e.target.value) || 16)}
                style={{ width: '80px', padding: '0.4rem', borderRadius: '6px', background: 'rgba(0,0,0,0.4)', border: '1px solid var(--border-color)', color: 'white', fontFamily: 'var(--font-mono)' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Add Model Modal */}
      <AddModelModal
        apiBase={apiBase}
        isOpen={isAddModelOpen}
        onClose={() => setIsAddModelOpen(false)}
        onSetupComplete={(alias) => {
          onRefresh();
          setTargetModel(alias);
        }}
      />

    </div>
  );
}
