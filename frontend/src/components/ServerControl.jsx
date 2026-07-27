import React, { useState, useEffect } from 'react';
import { Play, Square, RotateCw, Server, Cpu, HardDrive, Settings, AlertCircle, CheckCircle, Hammer, Wrench, ChevronDown, ChevronUp } from 'lucide-react';

export default function ServerControl({ apiBase, status, models, onRefresh }) {
  const [loading, setLoading] = useState(false);
  const [targetModel, setTargetModel] = useState(status?.model || 'gemma4');
  const [host, setHost] = useState(status?.host || '127.0.0.1');
  const [port, setPort] = useState(status?.port || 8000);
  const [legacy, setLegacy] = useState(status?.legacy || false);
  const [offline, setOffline] = useState(status?.offline || false);

  // Kernel Build States
  const [kernelStatus, setKernelStatus] = useState(null);
  const [showBuildOptions, setShowBuildOptions] = useState(false);
  const [noGemm, setNoGemm] = useState(false);
  const [maxBatch, setMaxBatch] = useState(16);
  const [buildingKernels, setBuildingKernels] = useState(false);

  useEffect(() => {
    fetchKernelStatus();
  }, []);

  const fetchKernelStatus = async () => {
    try {
      const res = await fetch(`${apiBase}/api/kernels/status`);
      const data = await res.json();
      setKernelStatus(data);
    } catch (e) {
      console.error("Failed to fetch kernel status:", e);
    }
  };

  const handleStart = async () => {
    setLoading(true);
    try {
      await fetch(`${apiBase}/api/control/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: targetModel, host, port, legacy, offline })
      });
      setTimeout(onRefresh, 1000);
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
      setTimeout(onRefresh, 1000);
    } catch (e) {
      alert(`Errore arresto server: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRestart = async () => {
    setLoading(true);
    try {
      await fetch(`${apiBase}/api/control/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: targetModel, host, port, legacy, offline })
      });
      setTimeout(onRefresh, 1500);
    } catch (e) {
      alert(`Errore riavvio server: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBuildKernels = async () => {
    if (!confirm(`Vuoi avviare la compilazione AIE dei kernel NPU per il modello '${targetModel}'? La procedura compila i file .xclbin e aggiorna kernels/build/manifest.json.`)) {
      return;
    }
    setBuildingKernels(true);
    try {
      const res = await fetch(`${apiBase}/api/control/build-kernels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: targetModel,
          no_gemm: noGemm,
          max_batch: maxBatch
        })
      });
      const data = await res.json();
      alert(`Compilazione kernel avviata per '${targetModel}'. Segui il progresso in tempo reale nella tab 'Log & Diagnostica'.`);
      setTimeout(fetchKernelStatus, 3000);
    } catch (e) {
      alert(`Errore avvio compilazione kernel: ${e}`);
    } finally {
      setBuildingKernels(false);
    }
  };

  const selectedModelObj = models.find(m => m.id === targetModel);
  const isManifestMismatch = kernelStatus?.manifest_exists &&
    kernelStatus.manifest_model_type &&
    selectedModelObj?.arch &&
    kernelStatus.manifest_model_type !== selectedModelObj.arch;

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Top Banner: Status & Actions */}
      <div className="glass-card" style={{ padding: '1.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <Server size={24} color="var(--accent-purple)" />
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700 }}>Inference Engine Controller</h2>
            
            {status?.status === 'running' && <span className="badge badge-success"><CheckCircle size={14} /> In Esecuzione</span>}
            {status?.status === 'stopped' && <span className="badge badge-danger"><AlertCircle size={14} /> Arrestato</span>}
            {status?.status === 'starting' && <span className="badge badge-warning"><RotateCw size={14} className="pulse-icon" /> In Avvio...</span>}
            {status?.status === 'building_kernels' && <span className="badge badge-warning"><Hammer size={14} className="pulse-icon" /> Compilazione Kernel...</span>}
          </div>
          
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {status?.is_running
              ? `Server attivo su http://${status.host}:${status.port} (PID: ${status.pid}, Uptime: ${status.uptime_seconds}s)`
              : "Il server NPU è attualmente spento. Configura le opzioni e clicca Avvia."}
          </p>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          {status?.is_running ? (
            <>
              <button className="btn-secondary" onClick={handleRestart} disabled={loading}>
                <RotateCw size={18} /> Riavvia
              </button>
              <button className="btn-danger" onClick={handleStop} disabled={loading}>
                <Square size={18} /> Arresta
              </button>
            </>
          ) : (
            <button className="btn-primary" onClick={handleStart} disabled={loading} style={{ background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}>
              <Play size={18} /> Avvia Server NPU
            </button>
          )}
        </div>
      </div>

      {/* Hardware Kernel Manager Banner */}
      <div className="glass-card" style={{ padding: '1.5rem', background: isManifestMismatch ? 'rgba(245, 158, 11, 0.08)' : 'rgba(23, 32, 54, 0.7)', border: isManifestMismatch ? '1px solid rgba(245, 158, 11, 0.4)' : '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
              <Hammer size={20} color={isManifestMismatch ? 'var(--accent-amber)' : 'var(--accent-cyan)'} />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Gestione Hardware Kernel NPU (`kernels/build`)</h3>
              
              {kernelStatus?.manifest_exists ? (
                <span className="badge badge-success"><CheckCircle size={14} /> Manifest OK ({kernelStatus.manifest_model_type})</span>
              ) : (
                <span className="badge badge-warning"><AlertCircle size={14} /> Manifest Assente</span>
              )}
            </div>

            <p style={{ fontSize: '0.9rem', color: isManifestMismatch ? '#fcd34d' : 'var(--text-muted)' }}>
              {isManifestMismatch ? (
                <>⚠️ Attenzione: I kernel compilati in <code>kernels/build</code> sono per l'architettura <strong>{kernelStatus.manifest_model_type}</strong>, mentre hai selezionato <strong>{targetModel}</strong> ({selectedModelObj?.arch}). Il server potrebbe fallire all'avvio. Clicca 'Compila Kernel NPU' per rigenerarli.</>
              ) : (
                <>I kernel hardware <code>.xclbin</code> vengono compilati via AOT per le matrici esatte del modello selezionato.</>
              )}
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <button className="btn-primary" onClick={handleBuildKernels} disabled={buildingKernels} style={{ background: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)' }}>
              <Wrench size={18} /> Compila / Ricrea Kernel per {targetModel}
            </button>

            <button className="btn-secondary" onClick={() => setShowBuildOptions(!showBuildOptions)}>
              {showBuildOptions ? <ChevronUp size={16} /> : <ChevronDown size={16} />} Opzioni Avanzate
            </button>
          </div>
        </div>

        {/* Advanced Kernel Options Accordion */}
        {showBuildOptions && (
          <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem' }}>
              <input type="checkbox" checked={noGemm} onChange={e => setNoGemm(e.target.checked)} />
              <span>--no-gemm (Salta forme GEMM per prefill più rapido)</span>
            </label>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem' }}>
              <span>--max-batch:</span>
              <input
                type="number"
                value={maxBatch}
                onChange={e => setMaxBatch(parseInt(e.target.value) || 16)}
                style={{ width: '70px', padding: '0.3rem', borderRadius: '6px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: 'white' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Grid: Settings & Model Switcher */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem' }}>
        
        {/* Settings Card */}
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Settings size={20} color="var(--accent-cyan)" /> Flag di Avvio & Rete
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                Seleziona Modello da Servire
              </label>
              <select
                value={targetModel}
                onChange={e => setTargetModel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.65rem',
                  borderRadius: '8px',
                  background: 'rgba(0,0,0,0.4)',
                  border: '1px solid var(--border-color)',
                  color: 'white',
                  fontSize: '0.95rem'
                }}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.id} ({m.arch})</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
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
                  Porta API
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
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                <input
                  type="checkbox"
                  checked={legacy}
                  onChange={e => setLegacy(e.target.checked)}
                />
                <span>Usa Server Legacy Python (Uvicorn / FastAPI)</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                <input
                  type="checkbox"
                  checked={offline}
                  onChange={e => setOffline(e.target.checked)}
                />
                <span>Modalità Offline (HF_HUB_OFFLINE=1)</span>
              </label>
            </div>
          </div>
        </div>

        {/* Installed Models Gallery */}
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <HardDrive size={20} color="var(--accent-purple)" /> Modelli Quantizzati Installati
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', maxHeight: '320px', overflowY: 'auto' }}>
            {models.map(m => (
              <div
                key={m.id}
                style={{
                  padding: '0.85rem 1rem',
                  borderRadius: '8px',
                  background: m.id === status?.model && status?.is_running ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255,255,255,0.03)',
                  border: m.id === status?.model && status?.is_running ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid var(--border-color)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: 'white', fontSize: '0.95rem' }}>{m.id}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Architettura: {m.arch} • {m.size_mb} MB
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <button
                    className="btn-secondary"
                    style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                    onClick={() => {
                      setTargetModel(m.id);
                      handleBuildKernels();
                    }}
                    title="Compila kernel hardware per questo modello"
                  >
                    <Hammer size={12} /> Kernel
                  </button>

                  {m.id === status?.model && status?.is_running ? (
                    <span className="badge badge-success">Attivo</span>
                  ) : (
                    <button
                      className="btn-secondary"
                      style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                      onClick={() => {
                        setTargetModel(m.id);
                        handleRestart();
                      }}
                    >
                      Carica
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
