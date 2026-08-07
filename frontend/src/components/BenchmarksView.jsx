import React, { useState, useEffect, useCallback } from 'react';
import { BarChart2, Zap, Activity, Cpu, Layers, HardDrive, AlertCircle, RefreshCw, CheckCircle, XCircle } from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';

export default function BenchmarksView({
  apiBase,
  status: initialStatus = null,
  models: initialModels = [],
  modelsLoading: initialModelsLoading = false,
  onRefresh
}) {
  const { t } = useTranslation();
  const [status, setStatus] = useState(initialStatus);
  const [models, setModels] = useState(initialModels);
  const [loading, setLoading] = useState(initialModelsLoading);
  const [error, setError] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (initialStatus) setStatus(initialStatus);
  }, [initialStatus]);

  useEffect(() => {
    if (initialModels && initialModels.length > 0) setModels(initialModels);
  }, [initialModels]);

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, modelsRes] = await Promise.all([
        fetch(`${apiBase}/api/status`),
        fetch(`${apiBase}/api/models`)
      ]);

      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setStatus(statusData);
      }
      if (modelsRes.ok) {
        const modelsData = await modelsRes.json();
        setModels(modelsData || []);
      }
      setError(false);
    } catch (e) {
      console.error("Failed to fetch benchmark metrics:", e);
      setError(true);
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const handleManualRefresh = async () => {
    setIsRefreshing(true);
    if (onRefresh) onRefresh();
    await fetchData();
    setIsRefreshing(false);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const formatSize = (mb) => {
    if (mb === undefined || mb === null || isNaN(mb)) return 'N/A';
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb} MB`;
  };

  const isModelRunning = status?.is_running || false;
  const activeModelId = status?.model;
  const tokPerSec = status?.tok_per_sec !== undefined ? Number(status.tok_per_sec).toFixed(2) : '0.00';
  const isLoaded = status?.is_loaded || false;
  const loadProgress = status?.load_progress || 0;
  const loadStep = status?.load_step || null;

  return (
    <div role="region" aria-label={t('benchmarks.title')} style={{ padding: '1.5rem 2rem', maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--text-main)' }}>
            <BarChart2 size={24} style={{ color: 'var(--accent-purple)' }} aria-hidden="true" />
            {t('benchmarks.title')}
          </h1>
          <p style={{ margin: '0.3rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
            {t('benchmarks.subtitle')}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-card)', padding: '0.35rem 0.75rem', borderRadius: '20px', border: '1px solid var(--border-color)' }}>
            <Zap size={13} color="var(--accent-amber)" aria-hidden="true" />
            {t('benchmarks.pollingNotice')}
          </span>

          <button
            className="btn-secondary"
            onClick={handleManualRefresh}
            disabled={isRefreshing}
            style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
            aria-label={t('benchmarks.refresh')}
          >
            <RefreshCw size={14} className={isRefreshing ? "pulse-icon" : ""} aria-hidden="true" />
            {t('benchmarks.refresh')}
          </button>
        </div>
      </div>

      {/* Error banner if connection fails */}
      {error && (
        <div
          role="alert"
          style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '12px',
            padding: '1rem 1.25rem',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.85rem',
            color: '#fca5a5'
          }}
        >
          <AlertCircle size={20} style={{ flexShrink: 0 }} />
          <div style={{ fontSize: '0.88rem' }}>
            {t('benchmarks.serverOffline')}
          </div>
        </div>
      )}

      {/* Section 1: Live Status & Benchmark Metrics Cards */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={18} color="var(--accent-green)" />
          {t('benchmarks.liveMetrics')}
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
          
          {/* Card 1: Active Model */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', boxShadow: 'var(--shadow-card)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', fontWeight: 600 }}>
              <span>{t('benchmarks.activeModel')}</span>
              <Cpu size={18} color="var(--accent-purple)" />
            </div>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-main)', wordBreak: 'break-word' }}>
              {activeModelId || t('benchmarks.noActiveModel')}
            </div>
            <div>
              {isModelRunning ? (
                <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>
                  <span className="pulse-icon">●</span> {t('benchmarks.running')}
                </span>
              ) : (
                <span className="badge badge-danger" style={{ fontSize: '0.75rem' }}>
                  ● {t('benchmarks.stopped')}
                </span>
              )}
            </div>
          </div>

          {/* Card 2: Load Progress */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', boxShadow: 'var(--shadow-card)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', fontWeight: 600 }}>
              <span>{t('benchmarks.loadStatus')}</span>
              <HardDrive size={18} color="var(--accent-amber)" />
            </div>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-main)' }}>
              {isLoaded
                ? t('benchmarks.loaded')
                : (isModelRunning
                    ? t('benchmarks.loading', { progress: loadProgress })
                    : t('benchmarks.notLoaded'))
              }
            </div>
            {isModelRunning && !isLoaded && (
              <div style={{ width: '100%', background: 'var(--bg-primary)', borderRadius: '6px', height: '6px', overflow: 'hidden', marginTop: '0.2rem' }}>
                <div style={{ width: `${loadProgress}%`, background: 'var(--accent-amber)', height: '100%', transition: 'width 0.3s ease' }} />
              </div>
            )}
            {loadStep && (
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.1rem', wordBreak: 'break-word' }}>
                {t('benchmarks.loadStep')}: {loadStep}
              </div>
            )}
          </div>

          {/* Card 3: Measured Decode Speed (tok/s) */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', boxShadow: 'var(--shadow-card)', borderLeft: '4px solid var(--accent-amber)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', fontWeight: 600 }}>
              <span>{t('benchmarks.decodeSpeed')}</span>
              <Zap size={18} color="var(--accent-amber)" />
            </div>
            <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-amber)', letterSpacing: '-0.02em' }}>
              {tokPerSec} <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>tok/s</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {t('benchmarks.speedDesc')}
            </div>
          </div>

        </div>
      </div>

      {/* Section 2: Model Directory Grid */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-main)', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={18} color="var(--accent-purple)" />
            {t('benchmarks.modelsTitle')}
          </h2>
          <span style={{ fontSize: '0.8rem', background: 'var(--bg-card)', padding: '0.25rem 0.65rem', borderRadius: '12px', border: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
            {models.length} {t('benchmarks.modelsCount')}
          </span>
        </div>

        {loading && models.length === 0 ? (
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <RefreshCw size={28} className="pulse-icon" style={{ marginBottom: '0.75rem' }} />
            <p style={{ margin: 0, fontSize: '0.95rem' }}>Loading models...</p>
          </div>
        ) : models.length === 0 ? (
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Layers size={36} style={{ opacity: 0.4, marginBottom: '0.75rem' }} />
            <p style={{ margin: 0, fontSize: '0.95rem' }}>{t('benchmarks.noModels')}</p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.25rem' }}>
            {models.map((model) => {
              const modelId = model.id || model.alias || 'unnamed-model';
              const isActive = model.active || (activeModelId === modelId && isModelRunning);

              return (
                <div
                  key={modelId}
                  style={{
                    background: 'var(--bg-card)',
                    border: isActive ? '1px solid var(--accent-green)' : '1px solid var(--border-color)',
                    borderRadius: '12px',
                    padding: '1.25rem',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '1rem',
                    boxShadow: 'var(--shadow-card)',
                    position: 'relative',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {/* Top line: Name and Active badge */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-main)', wordBreak: 'break-all' }}>
                        {modelId}
                      </div>
                      {model.alias && model.alias !== model.id && (
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                          Alias: {model.alias}
                        </div>
                      )}
                    </div>

                    {isActive ? (
                      <span className="badge badge-success" style={{ flexShrink: 0, fontSize: '0.72rem' }}>
                        <span className="pulse-icon">●</span> {t('benchmarks.activeBadge')}
                      </span>
                    ) : (
                      <span className="badge badge-secondary" style={{ flexShrink: 0, fontSize: '0.72rem', opacity: 0.7 }}>
                        {t('benchmarks.inactiveBadge')}
                      </span>
                    )}
                  </div>

                  {/* Metadata fields: Arch, Size, Config */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.82rem', background: 'var(--bg-primary)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        {t('benchmarks.arch')}
                      </span>
                      <strong style={{ color: 'var(--text-main)', fontSize: '0.88rem' }}>
                        {model.arch || 'Unknown'}
                      </strong>
                    </div>

                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        {t('benchmarks.size')}
                      </span>
                      <strong style={{ color: 'var(--text-main)', fontSize: '0.88rem' }}>
                        {formatSize(model.size_mb)}
                      </strong>
                    </div>

                    <div style={{ gridColumn: 'span 2' }}>
                      <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        {t('benchmarks.config')}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.15rem' }}>
                        {model.has_config ? (
                          <span style={{ color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '0.25rem', fontWeight: 600 }}>
                            <CheckCircle size={14} /> {t('benchmarks.configPresent')}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                            <XCircle size={14} /> {t('benchmarks.configMissing')}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* File Path */}
                  {model.path && (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      <span style={{ display: 'block', fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.15rem' }}>
                        {t('benchmarks.path')}:
                      </span>
                      <code style={{ background: 'var(--bg-primary)', padding: '0.25rem 0.5rem', borderRadius: '4px', border: '1px solid var(--border-color)', display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'monospace' }} title={model.path}>
                        {model.path}
                      </code>
                    </div>
                  )}

                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
