import React from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          aria-live="assertive"
          style={{
            minHeight: '400px',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
            background: 'var(--bg-primary)'
          }}
        >
          <div
            className="glass-card"
            style={{
              maxWidth: '560px',
              width: '100%',
              padding: '2.5rem',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '1.25rem',
              borderColor: 'rgba(239, 68, 68, 0.4)'
            }}
          >
            <div
              style={{
                width: '64px',
                height: '64px',
                borderRadius: '16px',
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 20px rgba(239, 68, 68, 0.2)'
              }}
            >
              <AlertTriangle size={32} color="#ef4444" />
            </div>

            <div>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-main)', marginBottom: '0.5rem' }}>
                Si è verificato un errore imprevisto
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: '1.6' }}>
                Un componente dell'interfaccia ha riscontrato un problema. L'applicazione è stata isolata per prevenire blocchi completi.
              </p>
            </div>

            {this.state.error && (
              <div
                style={{
                  width: '100%',
                  textAlign: 'left',
                  background: 'rgba(0, 0, 0, 0.4)',
                  borderRadius: '8px',
                  padding: '0.85rem 1rem',
                  border: '1px solid var(--border-color)',
                  fontSize: '0.82rem',
                  color: '#fca5a5',
                  fontFamily: 'var(--font-mono)',
                  maxHeight: '140px',
                  overflowY: 'auto'
                }}
              >
                <strong>Dettaglio Errore:</strong> {this.state.error.toString()}
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.85rem', flexWrap: 'wrap', justifyContent: 'center', width: '100%', marginTop: '0.5rem' }}>
              <button
                className="btn-primary"
                onClick={this.handleReset}
                aria-label="Riprova a caricare il componente"
              >
                <RefreshCw size={16} /> Riprova Componente
              </button>
              <button
                className="btn-secondary"
                onClick={this.handleReload}
                aria-label="Ricarica la pagina web"
              >
                <Home size={16} /> Ricarica Pagina
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
