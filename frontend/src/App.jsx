import React, { useState, useEffect, useCallback } from 'react';
import { MessageSquare, Server, Terminal, Zap, Settings, Moon, Sun, AlertTriangle, RefreshCw } from 'lucide-react';
import OnboardingWizard from './components/OnboardingWizard';
import ChatPlayground from './components/ChatPlayground';
import ServerControl from './components/ServerControl';
import LogsViewer from './components/LogsViewer';
import ErrorBoundary from './components/ErrorBoundary';
import LanguageSwitcher from './components/LanguageSwitcher';
import { useTranslation } from './i18n/I18nContext';

export default function App() {
  const { t } = useTranslation();

  const apiBase = window.location.origin.includes('5173')
    ? 'http://127.0.0.1:8080'
    : window.location.origin;

  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'control' | 'logs'
  const [status, setStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [showWizard, setShowWizard] = useState(false);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);

  // Theme state: 'dark' | 'light'
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('alveare_theme') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('alveare_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/status`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setStatus(data);
      setConnectionError(false);
      if (data.first_launch && loading) {
        setShowWizard(true);
      }
    } catch (e) {
      console.error("Failed to connect to Alveare Control Server:", e);
      setConnectionError(true);
    } finally {
      setLoading(false);
    }
  }, [apiBase, loading]);

  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const res = await fetch(`${apiBase}/api/models`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setModels(data || []);
      setConnectionError(false);
    } catch (e) {
      console.error("Failed to fetch models:", e);
      setConnectionError(true);
    } finally {
      setModelsLoading(false);
    }
  }, [apiBase]);

  const handleRetryConnection = async () => {
    setIsRetrying(true);
    await Promise.all([fetchStatus(), fetchModels()]);
    setIsRetrying(false);
  };

  useEffect(() => {
    fetchStatus();
    fetchModels();

    const interval = setInterval(() => {
      fetchStatus();
    }, 3000);

    return () => clearInterval(interval);
  }, [fetchStatus, fetchModels]);

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
      <nav
        role="navigation"
        aria-label="Navigazione Principale"
        style={{
          height: '64px',
          padding: '0 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          background: 'var(--navbar-bg)',
          backdropFilter: 'blur(12px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          zIndex: 50,
          flexWrap: 'wrap',
          gap: '0.5rem'
        }}
      >
        
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
            boxShadow: 'var(--shadow-glow)'
          }}>
            <Zap size={20} color="white" />
          </div>

          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', letterSpacing: '-0.02em', background: 'var(--gradient-brand)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              {t('nav.brandTitle')}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: '1' }}>
              {t('nav.brandSubtitle')}
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '0.5rem', background: 'var(--nav-tab-bg)', padding: '0.25rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          <button
            onClick={() => setActiveTab('chat')}
            aria-selected={activeTab === 'chat'}
            role="tab"
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
            <MessageSquare size={16} /> {t('nav.playground')}
          </button>

          <button
            onClick={() => setActiveTab('control')}
            aria-selected={activeTab === 'control'}
            role="tab"
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
            <Server size={16} /> {t('nav.controlPanel')}
            {models.length === 0 && !modelsLoading && (
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-amber)', display: 'inline-block' }} title={t('nav.noModelsBadge')} />
            )}
          </button>

          <button
            onClick={() => setActiveTab('logs')}
            aria-selected={activeTab === 'logs'}
            role="tab"
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
            <Terminal size={16} /> {t('nav.logs')}
          </button>
        </div>

        {/* Server Status & Controls Header Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {connectionError ? (
            <span className="badge badge-danger">
              ● {t('nav.serverUnreachable')}
            </span>
          ) : status?.is_running ? (
            <span className="badge badge-success">
              <span className="pulse-icon">●</span> {t('nav.serverActive', { model: status.model })}
            </span>
          ) : (
            <span className="badge badge-danger">
              ● {t('nav.serverStopped')}
            </span>
          )}

          {/* Language Switcher */}
          <LanguageSwitcher />

          {/* Theme Switcher Button */}
          <button
            className="btn-secondary"
            style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
            onClick={toggleTheme}
            title={theme === 'dark' ? t('nav.themeToggleLight') : t('nav.themeToggleDark')}
            aria-label={theme === 'dark' ? t('nav.themeToggleLight') : t('nav.themeToggleDark')}
          >
            {theme === 'dark' ? <Sun size={14} color="var(--accent-amber)" /> : <Moon size={14} color="var(--accent-purple)" />}
            <span style={{ fontSize: '0.78rem' }}>{theme === 'dark' ? t('nav.themeLight') : t('nav.themeDark')}</span>
          </button>

          <button
            className="btn-secondary"
            style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
            onClick={() => setShowWizard(true)}
            title={t('nav.reopenWizard')}
            aria-label={t('nav.reopenWizard')}
          >
            <Settings size={14} /> {t('nav.wizard')}
          </button>
        </div>

      </nav>

      {/* Connection Failure Resilient Banner */}
      {connectionError && (
        <div
          role="alert"
          aria-live="polite"
          style={{
            background: 'linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(245, 158, 11, 0.15) 100%)',
            borderBottom: '1px solid rgba(239, 68, 68, 0.3)',
            padding: '0.75rem 1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap',
            zIndex: 40
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.88rem', color: 'var(--text-main)' }}>
            <AlertTriangle size={20} color="#ef4444" style={{ flexShrink: 0 }} />
            <div>
              <strong>{t('connectionBanner.cannotConnect', { apiBase })}</strong>
              <span style={{ color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                {t('connectionBanner.verifyBackend')}
              </span>
            </div>
          </div>

          <button
            className="btn-secondary"
            onClick={handleRetryConnection}
            disabled={isRetrying}
            style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem', borderColor: 'rgba(239, 68, 68, 0.4)', color: '#fca5a5' }}
            aria-label={t('connectionBanner.retryConnection')}
          >
            <RefreshCw size={14} className={isRetrying ? "pulse-icon" : ""} />
            {isRetrying ? t('connectionBanner.connecting') : t('connectionBanner.retryConnection')}
          </button>
        </div>
      )}

      {/* Main View Body with ErrorBoundary protection */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
        <ErrorBoundary onReset={handleRetryConnection}>
          <div style={{ flex: 1, display: activeTab === 'chat' ? 'flex' : 'none', flexDirection: 'column', height: '100%' }}>
            <ChatPlayground
              apiBase={apiBase}
              activeModel={status?.model || (models.length > 0 ? models[0].id : t('nav.noModelsBadge'))}
              isServerRunning={status?.is_running || false}
              models={models}
              modelsLoading={modelsLoading}
              onNavigateToControl={() => setActiveTab('control')}
            />
          </div>

          <div style={{ flex: 1, display: activeTab === 'control' ? 'block' : 'none' }}>
            <ServerControl
              apiBase={apiBase}
              status={status}
              models={models}
              modelsLoading={modelsLoading}
              onRefresh={() => { fetchStatus(); fetchModels(); }}
            />
          </div>

          <div style={{ flex: 1, display: activeTab === 'logs' ? 'block' : 'none' }}>
            <LogsViewer apiBase={apiBase} />
          </div>
        </ErrorBoundary>
      </main>
    </div>
  );
}

