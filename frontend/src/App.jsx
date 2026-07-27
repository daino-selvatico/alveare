import React, { useState, useEffect } from 'react';
import { MessageSquare, Server, Terminal, Zap, Cpu, Settings, Play } from 'lucide-react';
import OnboardingWizard from './components/OnboardingWizard';
import ChatPlayground from './components/ChatPlayground';
import ServerControl from './components/ServerControl';
import LogsViewer from './components/LogsViewer';

export default function App() {
  const apiBase = window.location.origin.includes('5173')
    ? 'http://127.0.0.1:8080'
    : window.location.origin;

  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'control' | 'logs'
  const [status, setStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [showWizard, setShowWizard] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStatus();
    fetchModels();

    const interval = setInterval(() => {
      fetchStatus();
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${apiBase}/api/status`);
      const data = await res.json();
      setStatus(data);
      if (data.first_launch && loading) {
        setShowWizard(true);
      }
    } catch (e) {
      console.error("Failed to connect to Alveare Control Server:", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch(`${apiBase}/api/models`);
      const data = await res.json();
      setModels(data);
    } catch (e) {
      console.error("Failed to fetch models:", e);
    }
  };

  if (showWizard) {
    return (
      <OnboardingWizard
        apiBase={apiBase}
        onComplete={() => {
          setShowWizard(false);
          fetchStatus();
          fetchModels();
        }}
      />
    );
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>
      
      {/* Top Navbar */}
      <nav style={{
        height: '64px',
        padding: '0 1.5rem',
        borderBottom: '1px solid var(--border-color)',
        background: 'rgba(19, 27, 46, 0.7)',
        backdropFilter: 'blur(12px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        zIndex: 50
      }}>
        
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            background: 'var(--gradient-brand)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(139, 92, 246, 0.4)'
          }}>
            <Zap size={20} color="white" />
          </div>

          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', letterSpacing: '-0.02em', background: 'var(--gradient-brand)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              ALVEARE NPU
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: '1' }}>
              AMD Ryzen AI (XDNA2) Control Dashboard
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '0.5rem', background: 'rgba(0,0,0,0.3)', padding: '0.25rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          <button
            onClick={() => setActiveTab('chat')}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              border: 'none',
              background: activeTab === 'chat' ? 'var(--gradient-brand)' : 'transparent',
              color: activeTab === 'chat' ? 'white' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              transition: 'all 0.2s ease'
            }}
          >
            <MessageSquare size={16} /> Playground Chat
          </button>

          <button
            onClick={() => setActiveTab('control')}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              border: 'none',
              background: activeTab === 'control' ? 'var(--gradient-brand)' : 'transparent',
              color: activeTab === 'control' ? 'white' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              transition: 'all 0.2s ease'
            }}
          >
            <Server size={16} /> Control Panel & Modelli
          </button>

          <button
            onClick={() => setActiveTab('logs')}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              border: 'none',
              background: activeTab === 'logs' ? 'var(--gradient-brand)' : 'transparent',
              color: activeTab === 'logs' ? 'white' : 'var(--text-muted)',
              fontWeight: 600,
              fontSize: '0.85rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              transition: 'all 0.2s ease'
            }}
          >
            <Terminal size={16} /> Log & Diagnostica
          </button>
        </div>

        {/* Server Status Header Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          {status?.is_running ? (
            <span className="badge badge-success">
              <span className="pulse-icon">●</span> Server Attivo ({status.model})
            </span>
          ) : (
            <span className="badge badge-danger">
              ● Server Spento
            </span>
          )}

          <button
            className="btn-secondary"
            style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
            onClick={() => setShowWizard(true)}
            title="Riapri Setup Wizard"
          >
            <Settings size={14} /> Wizard
          </button>
        </div>

      </nav>

      {/* Main View Body */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'chat' && (
          <ChatPlayground
            apiBase={apiBase}
            activeModel={status?.model || 'gemma4'}
            isServerRunning={status?.is_running || false}
          />
        )}

        {activeTab === 'control' && (
          <ServerControl
            apiBase={apiBase}
            status={status}
            models={models}
            onRefresh={() => { fetchStatus(); fetchModels(); }}
          />
        )}

        {activeTab === 'logs' && (
          <LogsViewer apiBase={apiBase} />
        )}
      </main>
    </div>
  );
}
