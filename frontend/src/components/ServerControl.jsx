import React, { useState } from 'react';
import { Play, Square, RotateCw, Server, Cpu, HardDrive, Settings, AlertCircle, CheckCircle } from 'lucide-react';

export default function ServerControl({ apiBase, status, models, onRefresh }) {
  const [loading, setLoading] = useState(false);
  const [targetModel, setTargetModel] = useState(status?.model || 'gemma4');
  const [host, setHost] = useState(status?.host || '127.0.0.1');
  const [port, setPort] = useState(status?.port || 8000);
  const [legacy, setLegacy] = useState(status?.legacy || false);
  const [offline, setOffline] = useState(status?.offline || false);

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

                {m.id === status?.model && status?.is_running ? (
                  <span className="badge badge-success">Attivo</span>
                ) : (
                  <button
                    className="btn-secondary"
                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
                    onClick={() => {
                      setTargetModel(m.id);
                      handleRestart();
                    }}
                  >
                    Carica
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
