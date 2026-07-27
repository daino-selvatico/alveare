import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle, AlertTriangle, ArrowRight, Zap, Server, Sliders, Check } from 'lucide-react';

export default function OnboardingWizard({ onComplete, apiBase }) {
  const [step, setStep] = useState(1);
  const [npuStatus, setNpuStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Selected settings
  const [selectedModel, setSelectedModel] = useState('gemma4');
  const [host, setHost] = useState('127.0.0.1');
  const [port, setPort] = useState(8000);
  const [legacy, setLegacy] = useState(false);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    setLoading(true);
    try {
      const [npuRes, modelsRes] = await Promise.all([
        fetch(`${apiBase}/api/npu/check`),
        fetch(`${apiBase}/api/models`)
      ]);
      const npuData = await npuRes.json();
      const modelsData = await modelsRes.json();
      
      setNpuStatus(npuData);
      setModels(modelsData);
      if (modelsData.length > 0) {
        setSelectedModel(modelsData[0].id);
      }
    } catch (e) {
      console.error("Failed to load setup data:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleFinish = async () => {
    // 1. Save config first_launch = false
    await fetch(`${apiBase}/api/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        first_launch: false,
        default_model: selectedModel,
        host,
        port,
        legacy,
        offline
      })
    });

    // 2. Start server
    await fetch(`${apiBase}/api/control/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        host,
        port,
        legacy,
        offline
      })
    });

    onComplete();
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'radial-gradient(circle at 50% 30%, #1e1b4b 0%, #0b0f19 80%)',
      padding: '2rem'
    }}>
      <div className="glass-card" style={{ maxWidth: '650px', width: '100%', padding: '2.5rem' }}>
        
        {/* Header / Stepper */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.6rem',
            background: 'rgba(139, 92, 246, 0.15)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
            padding: '0.4rem 1rem',
            borderRadius: '9999px',
            color: '#a78bfa',
            fontWeight: 600,
            fontSize: '0.85rem',
            marginBottom: '1rem'
          }}>
            <Zap size={16} /> Alveare NPU Setup Wizard
          </div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: '700', marginBottom: '0.5rem' }}>
            {step === 1 && "Benvenuto in Alveare NPU"}
            {step === 2 && "Seleziona il Modello LLM"}
            {step === 3 && "Configura i Parametri di Avvio"}
            {step === 4 && "Setup Completato!"}
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            {step === 1 && "Verifica hardware dell'NPU AMD Ryzen AI (XDNA2)"}
            {step === 2 && "Scegli il modello quantizzato da caricare sull'NPU"}
            {step === 3 && "Imposta porta, indirizzo host ed opzioni del server"}
            {step === 4 && "Pronto per l'avvio del server e l'utilizzo dell'interfaccia"}
          </p>

          {/* Stepper Dots */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', marginTop: '1.5rem' }}>
            {[1, 2, 3, 4].map(s => (
              <div key={s} style={{
                width: '36px',
                height: '6px',
                borderRadius: '3px',
                background: step >= s ? 'var(--accent-purple)' : 'rgba(255,255,255,0.1)',
                transition: 'all 0.3s ease'
              }} />
            ))}
          </div>
        </div>

        {/* Step Content */}
        {step === 1 && (
          <div>
            <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '10px', padding: '1.25rem', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Cpu size={18} color="var(--accent-cyan)" /> Preflight Hardware NPU
              </h3>
              
              {loading ? (
                <p style={{ color: 'var(--text-muted)' }}>Analisi hardware in corso...</p>
              ) : npuStatus ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Dispositivo NPU (`/dev/accel/accel0`):</span>
                    {npuStatus.device_node ? (
                      <span className="badge badge-success"><Check size={14} /> Rilevato</span>
                    ) : (
                      <span className="badge badge-danger"><AlertTriangle size={14} /> Mancante</span>
                    )}
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Driver XRT SMI:</span>
                    {npuStatus.xrt_smi ? (
                      <span className="badge badge-success"><Check size={14} /> OK</span>
                    ) : (
                      <span className="badge badge-warning">Non installato</span>
                    )}
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Stack MLIR-AIE (PyXRT):</span>
                    {npuStatus.pyxrt_import ? (
                      <span className="badge badge-success"><Check size={14} /> Attivo</span>
                    ) : (
                      <span className="badge badge-warning">Usa fallback</span>
                    )}
                  </div>
                </div>
              ) : (
                <p style={{ color: 'var(--accent-red)' }}>Impossibile verificare lo stato NPU.</p>
              )}
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button className="btn-primary" onClick={() => setStep(2)}>
                Continua <ArrowRight size={18} />
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.5rem' }}>
              {models.map(m => (
                <div
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  style={{
                    padding: '1rem 1.25rem',
                    borderRadius: '10px',
                    border: selectedModel === m.id ? '2px solid var(--accent-purple)' : '1px solid var(--border-color)',
                    background: selectedModel === m.id ? 'rgba(139, 92, 246, 0.12)' : 'rgba(255,255,255,0.02)',
                    cursor: 'pointer',
                    display: 'flex',
                    justify: 'space-between',
                    alignItems: 'center',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '1.05rem', color: selectedModel === m.id ? '#c084fc' : 'white' }}>
                      {m.id}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Architettura: {m.arch} • Dimensione: {m.size_mb} MB
                    </div>
                  </div>
                  {selectedModel === m.id && <CheckCircle size={22} color="var(--accent-purple)" />}
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button className="btn-secondary" onClick={() => setStep(1)}>Indietro</button>
              <button className="btn-primary" onClick={() => setStep(3)}>
                Continua <ArrowRight size={18} />
              </button>
            </div>
          </div>
        )}

        {step === 3 && (
          <div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginBottom: '1.5rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem', color: 'var(--text-muted)' }}>
                  Indirizzo Host (Serve)
                </label>
                <input
                  type="text"
                  value={host}
                  onChange={e => setHost(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.65rem 0.9rem',
                    borderRadius: '8px',
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontFamily: 'var(--font-mono)'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem', color: 'var(--text-muted)' }}>
                  Porta Server HTTP (default: 8000)
                </label>
                <input
                  type="number"
                  value={port}
                  onChange={e => setPort(parseInt(e.target.value) || 8000)}
                  style={{
                    width: '100%',
                    padding: '0.65rem 0.9rem',
                    borderRadius: '8px',
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontFamily: 'var(--font-mono)'
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.5rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={legacy}
                    onChange={e => setLegacy(e.target.checked)}
                  />
                  <span>Server Legacy Python (fastapi)</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={offline}
                    onChange={e => setOffline(e.target.checked)}
                  />
                  <span>Modalità Offline (HF_HUB_OFFLINE)</span>
                </label>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button className="btn-secondary" onClick={() => setStep(2)}>Indietro</button>
              <button className="btn-primary" onClick={() => setStep(4)}>
                Riepilogo <ArrowRight size={18} />
              </button>
            </div>
          </div>
        )}

        {step === 4 && (
          <div>
            <div style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: '10px', padding: '1.25rem', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1.05rem', color: '#34d399', fontWeight: 600, marginBottom: '0.8rem' }}>
                Riepilogo Configurazione Iniziale
              </h3>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem', color: 'var(--text-main)', fontSize: '0.95rem' }}>
                <li>• <strong>Modello selezionato:</strong> {selectedModel}</li>
                <li>• <strong>Endpoint API:</strong> http://{host}:{port}/v1</li>
                <li>• <strong>Runtime Engine:</strong> {legacy ? "Legacy Python (Uvicorn)" : "Native C++ Server (XRT Direct)"}</li>
                <li>• <strong>Offline:</strong> {offline ? "Sì" : "No"}</li>
              </ul>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <button className="btn-secondary" onClick={() => setStep(3)}>Indietro</button>
              <button className="btn-primary" onClick={handleFinish} style={{ background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}>
                Completa Setup e Avvia Server <Zap size={18} />
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
