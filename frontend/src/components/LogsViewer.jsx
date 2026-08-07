import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Terminal, Cpu, RefreshCw } from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';

export default function LogsViewer({ apiBase }) {
  const { t } = useTranslation();
  const [logs, setLogs] = useState([]);
  const [npu, setNpu] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const logEndRef = useRef(null);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/logs`);
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch (e) {
      console.error("Failed to fetch logs:", e);
    }
  }, [apiBase]);

  const fetchNpu = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/npu/check`);
      if (res.ok) {
        const data = await res.json();
        setNpu(data);
      }
    } catch (e) {
      console.error("Failed to fetch NPU status:", e);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchLogs();
    fetchNpu();
    const interval = setInterval(() => {
      fetchLogs();
    }, 2000);
    return () => clearInterval(interval);
  }, [fetchLogs, fetchNpu]);

  useEffect(() => {
    if (autoScroll) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* NPU Health Card */}
      <div className="glass-card" style={{ padding: '1.25rem' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={20} color="var(--accent-cyan)" /> {t('logsViewer.npuDiagnosticsTitle')}
        </h3>

        {npu ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
            <div style={{ padding: '0.85rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{t('logsViewer.deviceNodeLabel')}</div>
              <div style={{ fontWeight: 600, marginTop: '0.2rem', color: npu.device_node ? '#34d399' : '#fca5a5' }}>
                {npu.device_node ? t('logsViewer.deviceNodeOk') : t('logsViewer.deviceNodeNotFound')}
              </div>
            </div>

            <div style={{ padding: '0.85rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{t('logsViewer.xrtSmiLabel')}</div>
              <div style={{ fontWeight: 600, marginTop: '0.2rem', color: npu.xrt_smi ? '#34d399' : '#fca5a5' }}>
                {npu.xrt_smi ? t('logsViewer.xrtSmiInstalled') : t('logsViewer.xrtSmiMissing')}
              </div>
            </div>

            <div style={{ padding: '0.85rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{t('logsViewer.pyxrtLabel')}</div>
              <div style={{ fontWeight: 600, marginTop: '0.2rem', color: npu.pyxrt_import ? '#34d399' : '#fca5a5' }}>
                {npu.pyxrt_import ? t('logsViewer.pyxrtAvailable') : t('logsViewer.pyxrtNotImportable')}
              </div>
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>{t('logsViewer.loadingDiagnostics')}</p>
        )}
      </div>

      {/* Terminal Console Log View */}
      <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Terminal size={20} color="var(--accent-purple)" /> {t('logsViewer.realtimeServerLogs')}
          </h3>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1.rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', cursor: 'pointer' }}>
              <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
              <span>{t('logsViewer.autoScroll')}</span>
            </label>
            <button className="btn-secondary" onClick={fetchLogs} style={{ padding: '0.35rem 0.75rem', fontSize: '0.85rem' }} aria-label={t('logsViewer.refresh')}>
              <RefreshCw size={14} /> {t('logsViewer.refresh')}
            </button>
          </div>
        </div>

        <div style={{
          background: '#090d16',
          borderRadius: '8px',
          padding: '1rem',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.85rem',
          color: '#38bdf8',
          height: '420px',
          overflowY: 'auto',
          lineHeight: '1.5',
          border: '1px solid rgba(255,255,255,0.05)'
        }}>
          {logs.length === 0 ? (
            <div style={{ color: 'var(--text-subtle)' }}>{t('logsViewer.noLogsAvailable')}</div>
          ) : (
            logs.map((line, idx) => (
              <div key={idx} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {line}
              </div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      </div>

    </div>
  );
}

