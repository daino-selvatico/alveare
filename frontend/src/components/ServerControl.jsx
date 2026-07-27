import React, { useState, useEffect } from 'react';
import { Play, Square, RotateCw, Server, Cpu, HardDrive, Settings, AlertCircle, CheckCircle, Hammer, Wrench, ChevronDown, ChevronUp, Layers } from 'lucide-react';

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
  const [buildingModel, setBuildingModel] = useState(null);

  useEffect(() => {
    fetchKernelStatus();
  }, [status?.model]);

  const fetchKernelStatus = async () => {
    try {
      const res = await fetch(`${apiBase}/api/kernels/status`);
      const data = await res.json();
      setKernelStatus(data);
    } catch (e) {
      console.error("Failed to fetch kernel status:", e);
    }
  };

  const handleStart = async (modelToStart) => {
    const selected = modelToStart || targetModel;
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

  const activeModelObj = models.find(m => m.id === status?.model);

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* 1. HERO SERVER STATUS BANNER */}
      <div className="glass-card" style={{ padding: '1.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <Server size={26} color="var(--accent-purple)" />
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700 }}>Inference Engine Controller</h2>
            
            {status?.status === 'running' && <span className="badge badge-success"><CheckCircle size={14} /> In Esecuzione ({status.model})</span>}
            {status?.status === 'stopped' && <span className="badge badge-danger"><AlertCircle size={14} /> Server Spento</span>}
            {status?.status === 'starting' && <span className="badge badge-warning"><RotateCw size={14} className="pulse-icon" /> In Avvio...</span>}
            {status?.status === 'building_kernels' && <span className="badge badge-warning"><Hammer size={14} className="pulse-icon" /> Compilazione Kernel NPU in corso...</span>}
          </div>
          
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {status?.is_running
              ? `Server attivo su http://${status.host}:${status.port} (PID: ${status.pid}, Uptime: ${status.uptime_seconds}s)`
              : "Seleziona un modello dalla scheda sottostante e clicca 'Carica & Avvia'."}
          </p>
        </div>

        {/* Global Action Controls */}
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          {status?.is_running ? (
            <>
              <button className="btn-secondary" onClick={() => handleRestart(targetModel)} disabled={loading}>
                <RotateCw size={18} /> Riavvia
              </button>
              <button className="btn-danger" onClick={handleStop} disabled={loading}>
                <Square size={18} /> Arresta
              </button>
            </>
          ) : (
            <button className="btn-primary" onClick={() => handleStart(targetModel)} disabled={loading} style={{ background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}>
              <Play size={18} /> Avvia Server NPU ({targetModel})
            </button>
          )}
        </div>
      </div>

      {/* 2. UNIFIED MODEL GALLERY */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'white' }}>
            <HardDrive size={22} color="var(--accent-cyan)" /> Modelli Quantizzati Disponibili
          </h3>

          {kernelStatus && (
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Kernel salvati in <code>kernels/build/&lt;modello&gt;</code> per avvio immediato
            </span>
          )}
        </div>

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
                    <h4 style={{ fontSize: '1.15rem', fontWeight: 700, color: isSelected ? '#c084fc' : 'white' }}>
                      {m.id}
                    </h4>
                    <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                      Architettura: <strong>{m.arch}</strong> • Dimensione: {m.size_mb} MB
                    </span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.3rem' }}>
                    {isServed && <span className="badge badge-success"><CheckCircle size={12} /> IN ESECUZIONE</span>}
                    {isSelected && !isServed && <span className="badge" style={{ background: 'rgba(139,92,246,0.2)', color: '#c084fc', border: '1px solid rgba(139,92,246,0.4)' }}>SELEZIONATO</span>}
                  </div>
                </div>

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
                      isServed ? handleRestart(m.id) : handleStart(m.id);
                    }}
                    disabled={loading}
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
                  >
                    <Hammer size={14} /> {buildingModel === m.id ? "Compilazione..." : "Compila Kernel"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. SETTINGS & OPTIONS PANEL */}
      <div className="glass-card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Settings size={20} color="var(--accent-cyan)" /> Configurazione Server & Opzioni Avanzate Kernel
          </h3>

          <button className="btn-secondary" onClick={() => setShowBuildOptions(!showBuildOptions)} style={{ padding: '0.4rem 0.75rem', fontSize: '0.85rem' }}>
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

    </div>
  );
}
