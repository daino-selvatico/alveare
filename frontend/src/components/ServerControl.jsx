import React, { useState, useEffect, useCallback } from 'react';
import { Play, Square, RotateCw, Server, HardDrive, Settings, AlertCircle, CheckCircle, Hammer, ChevronDown, ChevronUp, Plus, Trash2 } from 'lucide-react';
import AddModelModal from './AddModelModal';
import { useTranslation } from '../i18n/I18nContext';

export default function ServerControl({ apiBase, status, models = [], modelsLoading = false, onRefresh }) {
  const { t } = useTranslation();
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
      alert(t('serverControl.deleteRunningModelAlert', { modelId }));
      return;
    }
    if (!confirm(t('serverControl.deleteConfirm', { modelId }))) {
      return;
    }
    try {
      const res = await fetch(`${apiBase}/api/models/${encodeURIComponent(modelId)}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        alert(t('serverControl.deleteSuccess', { modelId }));
        onRefresh();
        fetchKernelStatus();
      } else {
        const err = await res.json();
        alert(t('serverControl.deleteError', { error: err.detail || 'Failed' }));
      }
    } catch (e) {
      alert(t('serverControl.deleteError', { error: e }));
    }
  };

  const handleStart = async (modelToStart) => {
    const selected = modelToStart || targetModel;
    if (!selected) {
      alert(t('serverControl.selectModelFirst'));
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
      alert(t('serverControl.startError', { error: e }));
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
      alert(t('serverControl.stopError', { error: e }));
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
      alert(t('serverControl.restartError', { error: e }));
    } finally {
      setLoading(false);
    }
  };

  const handleBuildKernels = async (modelId) => {
    if (!confirm(t('serverControl.buildKernelConfirm', { modelId }))) {
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
      alert(t('serverControl.buildKernelStarted', { modelId }));
      setTimeout(fetchKernelStatus, 3000);
    } catch (e) {
      alert(t('serverControl.buildKernelError', { error: e }));
    } finally {
      setBuildingModel(null);
    }
  };

  return (
    <div role="region" aria-label={t('serverControl.engineController')} style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* 1. HERO SERVER STATUS BANNER */}
      <div className="glass-card" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <Server size={26} color="var(--accent-purple)" aria-hidden="true" />
              <h2 style={{ fontSize: '1.4rem', fontWeight: 700 }}>{t('serverControl.engineController')}</h2>
              
              {status?.status === 'running' && status?.is_loaded && (
                <>
                  <span className="badge badge-success"><CheckCircle size={14} aria-hidden="true" /> {t('serverControl.statusRunning', { model: status.model })}</span>
                  <span className="badge" style={{
                    background: (status.model === 'whisper-base' || status.model === 'sensevoice') ? 'rgba(59, 130, 246, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                    color: (status.model === 'whisper-base' || status.model === 'sensevoice') ? '#60a5fa' : '#34d399',
                    border: (status.model === 'whisper-base' || status.model === 'sensevoice') ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid rgba(16, 185, 129, 0.4)',
                    fontWeight: 700
                  }}>
                    {(status.model === 'whisper-base' || status.model === 'sensevoice') ? '🖥️ CPU (Whisper STT)' : '⚡ NPU (XDNA2 Native)'}
                  </span>
                </>
              )}
              {(status?.status === 'starting' || (status?.is_running && !status?.is_loaded)) && (
                <span className="badge badge-warning"><RotateCw size={14} className="pulse-icon" aria-hidden="true" /> {t('serverControl.statusLoading', { progress: status?.load_progress || 0 })}</span>
              )}
              {status?.status === 'stopped' && <span className="badge badge-danger"><AlertCircle size={14} aria-hidden="true" /> {t('serverControl.statusStopped')}</span>}
              {status?.status === 'error' && <span className="badge badge-danger"><AlertCircle size={14} aria-hidden="true" /> {t('serverControl.statusError')}</span>}
              {status?.status === 'building_kernels' && <span className="badge badge-warning"><Hammer size={14} className="pulse-icon" aria-hidden="true" /> {t('serverControl.statusBuildingKernels')}</span>}
            </div>
            
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              {status?.is_running
                ? t('serverControl.serverActiveOn', { host: status.host, port: status.port, pid: status.pid, uptime: status.uptime_seconds })
                : models.length > 0
                ? t('serverControl.selectModelToStart')
                : t('serverControl.noModelsFound')}
            </p>
          </div>

          {/* Global Action Controls */}
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            {status?.is_running ? (
              <>
                <button className="btn-secondary" onClick={() => handleRestart(targetModel)} disabled={loading} aria-label={t('serverControl.restartNpu')}>
                  <RotateCw size={18} className={loading ? "pulse-icon" : ""} aria-hidden="true" /> {t('serverControl.restart')}
                </button>
                <button className="btn-danger" onClick={handleStop} disabled={loading} aria-label={t('serverControl.stopNpu')}>
                  <Square size={18} aria-hidden="true" /> {t('serverControl.stop')}
                </button>
              </>
            ) : (
              <button
                className="btn-primary"
                onClick={() => handleStart(targetModel)}
                disabled={loading || models.length === 0}
                style={{ background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}
                aria-label={t('serverControl.startNpu')}
              >
                <Play size={18} className={loading ? "pulse-icon" : ""} aria-hidden="true" />
                {loading ? t('serverControl.starting') : targetModel ? t('serverControl.startServerModel', { model: targetModel }) : t('serverControl.startServer')}
              </button>
            )}
          </div>
        </div>

        {/* Model Load Progress Bar */}
        {(status?.status === 'starting' || (status?.is_running && !status?.is_loaded)) && (
          <div
            role="status"
            aria-live="polite"
            style={{
              background: 'rgba(0,0,0,0.3)',
              borderRadius: '10px',
              padding: '1rem',
              border: '1px solid rgba(245, 158, 11, 0.3)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: 'white', marginBottom: '0.4rem' }}>
              <span>{status?.load_step || t('serverControl.loadingWeightsStep')}</span>
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
          <div
            role="alert"
            style={{
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.4)',
              borderRadius: '10px',
              padding: '1rem 1.25rem',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.85rem'
            }}
          >
            <AlertCircle size={22} color="#ef4444" style={{ marginTop: '0.1rem', flexShrink: 0 }} aria-hidden="true" />
            <div style={{ flex: 1 }}>
              <div style={{ color: '#ef4444', fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.25rem' }}>
                {t('serverControl.executionErrorTitle')}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap', maxHeight: '120px', overflowY: 'auto' }}>
                {status?.last_error || t('serverControl.serverStoppedUnexpectedly')}
              </div>
            </div>
            <button
              className="btn-secondary"
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem', borderColor: 'rgba(239, 68, 68, 0.4)', color: '#ef4444' }}
              onClick={() => handleStart(targetModel)}
              aria-label={t('serverControl.retry')}
            >
              {t('serverControl.retry')}
            </button>
          </div>
        )}
      </div>

      {/* 2. UNIFIED MODEL GALLERY / EMPTY ONBOARDING STATE */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-main)' }}>
            <HardDrive size={22} color="var(--accent-cyan)" /> {t('serverControl.availableModels')}
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
              aria-label={t('serverControl.addModel')}
            >
              <Plus size={16} /> {t('serverControl.addModel')}
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
                {t('serverControl.noModelsTitle')}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: '1.6' }}>
                {t('serverControl.noModelsDesc')}
              </p>
            </div>

            <button
              onClick={() => setIsAddModelOpen(true)}
              className="btn-primary"
              style={{ padding: '0.75rem 1.5rem', fontSize: '0.95rem' }}
              aria-label={t('serverControl.downloadOrImportFirst')}
            >
              <Plus size={18} /> {t('serverControl.downloadOrImportFirst')}
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
                        {t('serverControl.arch', { arch: m.arch, size: m.size_mb })}
                      </span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.3rem' }}>
                      {isServed && status?.is_loaded && <span className="badge badge-success"><CheckCircle size={12} /> {t('serverControl.inExecution')}</span>}
                      {isServed && !status?.is_loaded && <span className="badge badge-warning"><RotateCw size={12} className="pulse-icon" /> {t('serverControl.loadingBadge', { progress: status?.load_progress || 0 })}</span>}
                      {isSelected && !isServed && <span className="badge" style={{ background: 'rgba(139,92,246,0.2)', color: '#c084fc', border: '1px solid rgba(139,92,246,0.4)' }}>{t('serverControl.selectedBadge')}</span>}
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
                      <span>{t('serverControl.measuredSpeed')}</span>
                      <span>⚡ {status.tok_per_sec} tok/s</span>
                    </div>
                  )}

                  {/* Kernel Status Indicator */}
                  <div style={{ fontSize: '0.82rem', padding: '0.5rem 0.75rem', borderRadius: '6px', background: 'rgba(0,0,0,0.35)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>{m.task === 'speech-to-text' || m.arch === 'sensevoice' || m.arch === 'whisper' ? 'Motore di Esecuzione' : t('serverControl.hardwareKernels')}</span>
                    {m.task === 'speech-to-text' || m.arch === 'sensevoice' || m.arch === 'whisper' ? (
                      <span style={{ color: '#06b6d4', fontWeight: 600 }}>🎙️ CPU (Whisper STT)</span>
                    ) : isKernelMatching ? (
                      <span style={{ color: '#34d399', fontWeight: 600 }}>
                        {t('serverControl.compiledCount', { count: mKernel.kernels_count })}
                      </span>
                    ) : (
                      <span style={{ color: '#fbbf24', fontWeight: 600 }}>
                        {t('serverControl.notCompiled')}
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
                      {isServed ? t('serverControl.restart') : t('serverControl.loadAndStart')}
                    </button>

                    {m.task !== 'speech-to-text' && m.arch !== 'sensevoice' && (
                      <button
                        className="btn-secondary"
                        style={{ padding: '0.5rem 0.75rem', fontSize: '0.85rem', color: '#c084fc', borderColor: 'rgba(139,92,246,0.4)' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setTargetModel(m.id);
                          handleBuildKernels(m.id);
                        }}
                        disabled={buildingModel === m.id}
                        title={t('serverControl.compileKernelTitle')}
                        aria-label={`Compila kernel NPU per ${m.id}`}
                      >
                        <Hammer size={14} /> {buildingModel === m.id ? t('serverControl.compiling') : t('serverControl.compileKernel')}
                      </button>
                    )}

                    <button
                      className="btn-danger"
                      style={{ padding: '0.5rem 0.65rem', fontSize: '0.85rem' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteModel(m.id);
                      }}
                      disabled={loading || (status?.is_running && status?.model === m.id)}
                      title={t('serverControl.deleteModelTitle')}
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
            <Settings size={20} color="var(--accent-cyan)" /> {t('serverControl.serverConfig')}
          </h3>

          <button className="btn-secondary" onClick={() => setShowBuildOptions(!showBuildOptions)} style={{ padding: '0.4rem 0.75rem', fontSize: '0.85rem' }} aria-expanded={showBuildOptions}>
            {showBuildOptions ? <ChevronUp size={16} /> : <ChevronDown size={16} />} {t('serverControl.advancedCompilationOptions')}
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.25rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              {t('serverControl.hostAddress')}
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
              {t('serverControl.httpPort')}
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
              <span>{t('serverControl.legacyPythonServer')}</span>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.88rem' }}>
              <input type="checkbox" checked={offline} onChange={e => setOffline(e.target.checked)} />
              <span>{t('serverControl.offlineMode')}</span>
            </label>
          </div>
        </div>

        {/* Expandable Advanced Kernel Compilation Options */}
        {showBuildOptions && (
          <div style={{ marginTop: '1.25rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '2rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.88rem' }}>
              <input type="checkbox" checked={noGemm} onChange={e => setNoGemm(e.target.checked)} />
              <span>{t('serverControl.noGemmOption')}</span>
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
