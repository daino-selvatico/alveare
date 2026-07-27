import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Cpu, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';

export default function LogsViewer({ apiBase }) {
  const [logs, setLogs] = useState([]);
  const [npu, setNpu] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const logEndRef = useRef(null);

  useEffect(() => {
    fetchLogs();
    fetchNpu();
    const interval = setInterval(() => {
      fetchLogs();
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (autoScroll) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const fetchLogs = async () => {
    try {
      const res = await fetch(`${apiBase}/api/logs`);
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (e) {
      console.error("Failed to fetch logs:", e);
    }
  };

  const fetchNpu = async () => {
    try {
      const res = await fetch(`${apiBase}/api/npu/check`);
      const data = await res.json();
      setNpu(data);
    } catch (e) {
      console.error("Failed to fetch NPU status:", e);
    }
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* NPU Health Card */}
      <div className="glass-card" style={{ padding: '1.25rem' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={20} color="var(--accent-cyan)" /> Diagnostica Hardware AMD Ryzen AI (XDNA2)
        </h3>

        {npu ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
            <div style={{ padding: '0.85rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Nodo Dispositivo NPU</div>
              <div style={{ fontWeight: 600, marginTop: '0.2rem', color: npu.device_node ? '#34d399' : '#fca5a5' }}>
                {npu.device_node ? "✓ /dev/accel/accel0 OK" : "✗ Non trovato"}
              </div>
            </div>

            <div style={{ padding: '0.85rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Driver XRT SMI</div>
              <div style={{ fontWeight: 600, marginTop: '0.2rem', color: npu.xrt_smi ? '#34d399' : '#fca5a5' }}>
                {npu.xrt_smi ? "✓ xrt-smi installato" : "✗ Mancante"}
              </div>
            </div>

            <div style={{ padding: '0.85rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>PyXRT Python Bindings</div>
              <div style={{ fontWeight: 600, marginTop: '0.2rem', color: npu.pyxrt_import ? '#34d399' : '#fca5a5' }}>
                {npu.pyxrt_import ? "✓ pyxrt disponibile" : "✗ Non importabile"}
              </div>
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>Caricamento diagnostica...</p>
        )}
      </div>

      {/* Terminal Console Log View */}
      <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Terminal size={20} color="var(--accent-purple)" /> Log in Realtime del Server
          </h3>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', cursor: 'pointer' }}>
              <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
              <span>Auto-scroll</span>
            </label>
            <button className="btn-secondary" onClick={fetchLogs} style={{ padding: '0.35rem 0.75rem', fontSize: '0.85rem' }}>
              <RefreshCw size={14} /> Aggiorna
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
            <div style={{ color: 'var(--text-subtle)' }}>Nessun log disponibile al momento.</div>
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
