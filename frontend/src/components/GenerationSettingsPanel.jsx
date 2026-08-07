import React, { useState } from 'react';
import {
  Sliders,
  Brain,
  ToggleLeft,
  ToggleRight,
  RotateCcw,
  Save,
  X,
  FileText,
  Thermometer,
  Layers,
  Check
} from 'lucide-react';
import { DEFAULT_SETTINGS } from '../utils/chatStorage';

const SYSTEM_PROMPT_PRESETS = [
  {
    label: '🤖 Generale',
    prompt: 'Sei un assistente AI esperto ed utile.'
  },
  {
    label: '💻 Programmatore Senior',
    prompt: 'Sei uno sviluppatore senior esperto in architettura software, refactoring, TypeScript, Python e C++. Fornisci risposte pratiche, ben strutturate e codice privo di bug.'
  },
  {
    label: '⚡ Risposte Sintetiche',
    prompt: 'Rispondi in modo estremamente sintetico, diretto ed essenziale, senza preamboli o spiegazioni superflue.'
  },
  {
    label: '🔍 Code Reviewer',
    prompt: 'Sei un esperto di analisi del codice e sicurezza. Ispeziona il codice fornito per identificare bug, vulnerabilità di sicurezza, colli di bottiglia nelle prestazioni e suggerisci ottimizzazioni precise.'
  }
];

export default function GenerationSettingsPanel({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
  onSaveAsGlobalDefaults,
  onResetToDefaults
}) {
  const [saveToast, setSaveToast] = useState(false);

  if (!isOpen) return null;

  const currentSettings = { ...DEFAULT_SETTINGS, ...settings };

  const handleChange = (key, value) => {
    onUpdateSettings({
      ...currentSettings,
      [key]: value
    });
  };

  const handleSaveDefaults = () => {
    onSaveAsGlobalDefaults(currentSettings);
    setSaveToast(true);
    setTimeout(() => setSaveToast(false), 2000);
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        justifyContent: 'flex-end',
        background: 'rgba(0, 0, 0, 0.4)',
        backdropFilter: 'blur(4px)',
        animation: 'fadeIn 0.2s ease-out'
      }}
      onClick={onClose}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: '420px',
          maxWidth: '90vw',
          height: '100%',
          background: 'var(--bg-secondary)',
          borderLeft: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '-8px 0 32px rgba(0, 0, 0, 0.5)',
          overflow: 'hidden'
        }}
      >
        {/* Panel Header */}
        <div style={{
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(0, 0, 0, 0.15)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <div style={{
              padding: '0.45rem',
              borderRadius: '10px',
              background: 'rgba(139, 92, 246, 0.15)',
              color: 'var(--accent-purple)',
              display: 'flex'
            }}>
              <Sliders size={20} />
            </div>
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                Impostazioni Generazione
              </h3>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
                Parametri modello e prompt di sistema
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '0.4rem',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            className="btn-icon-hover"
            title="Chiudi pannello"
          >
            <X size={20} />
          </button>
        </div>

        {/* Panel Content (Scrollable) */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem'
        }}>

          {/* Section 1: System Prompt */}
          <div className="glass-card" style={{ padding: '1.1rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <FileText size={16} color="var(--accent-purple)" /> Prompt di Sistema
              </label>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                {currentSettings.systemPrompt.length} caratteri
              </span>
            </div>

            <textarea
              value={currentSettings.systemPrompt}
              onChange={e => handleChange('systemPrompt', e.target.value)}
              rows={4}
              placeholder="Inserisci qui le istruzioni o la personalità per il modello..."
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '8px',
                background: 'rgba(0, 0, 0, 0.3)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-main)',
                fontSize: '0.85rem',
                lineHeight: '1.5',
                resize: 'vertical',
                outline: 'none',
                fontFamily: 'var(--font-sans)'
              }}
            />

            {/* Presets */}
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.4rem', fontWeight: 600 }}>
                Preset Rapidi:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {SYSTEM_PROMPT_PRESETS.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleChange('systemPrompt', preset.prompt)}
                    style={{
                      padding: '0.3rem 0.6rem',
                      fontSize: '0.75rem',
                      borderRadius: '6px',
                      border: '1px solid var(--border-color)',
                      background: currentSettings.systemPrompt === preset.prompt ? 'rgba(139, 92, 246, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                      color: currentSettings.systemPrompt === preset.prompt ? 'white' : 'var(--text-muted)',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Section 2: Thinking (Chain-of-Thought) */}
          <div
            className="glass-card"
            style={{
              padding: '1.1rem',
              borderColor: currentSettings.enableThinking ? 'rgba(139, 92, 246, 0.3)' : 'var(--border-color)',
              background: currentSettings.enableThinking ? 'rgba(139, 92, 246, 0.08)' : 'var(--bg-card)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <div style={{
                  padding: '0.4rem',
                  borderRadius: '8px',
                  background: currentSettings.enableThinking ? 'rgba(139, 92, 246, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                  color: currentSettings.enableThinking ? 'var(--accent-purple)' : 'var(--text-muted)'
                }}>
                  <Brain size={18} />
                </div>
                <div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    Thinking (Chain-of-Thought)
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                    {currentSettings.enableThinking ? 'Ragionamento visibile attivo' : 'Risposte dirette senza CoT'}
                  </div>
                </div>
              </div>

              <button
                onClick={() => handleChange('enableThinking', !currentSettings.enableThinking)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: currentSettings.enableThinking ? 'var(--accent-purple)' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  padding: 0
                }}
                title="Attiva/Disattiva Thinking"
              >
                {currentSettings.enableThinking ? <ToggleRight size={32} /> : <ToggleLeft size={32} />}
              </button>
            </div>
          </div>

          {/* Section 3: Sampling Parameters */}
          <div className="glass-card" style={{ padding: '1.1rem', display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <Thermometer size={16} color="var(--accent-cyan)" /> Parametri di Campionamento
            </div>

            {/* Temperature */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Temperatura (Temperature)
                </span>
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.05"
                  value={currentSettings.temperature}
                  onChange={e => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val)) handleChange('temperature', Math.max(0, Math.min(2, val)));
                  }}
                  style={{
                    width: '64px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '6px',
                    background: 'rgba(0, 0, 0, 0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontWeight: 700,
                    fontSize: '0.82rem',
                    textAlign: 'right'
                  }}
                />
              </div>

              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={currentSettings.temperature}
                onChange={e => handleChange('temperature', parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-purple)' }}
              />

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                <span>0.0 (Preciso)</span>
                <span>0.7 (Bilanciato)</span>
                <span>1.5+ (Creativo)</span>
              </div>
            </div>

            {/* Top-P */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Top-P (Nucleus Sampling)
                </span>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={currentSettings.topP}
                  onChange={e => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val)) handleChange('topP', Math.max(0, Math.min(1, val)));
                  }}
                  style={{
                    width: '64px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '6px',
                    background: 'rgba(0, 0, 0, 0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontWeight: 700,
                    fontSize: '0.82rem',
                    textAlign: 'right'
                  }}
                />
              </div>

              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={currentSettings.topP}
                onChange={e => handleChange('topP', parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-cyan)' }}
              />

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                <span>0.1 (Focalizzato)</span>
                <span>0.9 (Standard)</span>
                <span>1.0 (Tutti i token)</span>
              </div>
            </div>

            {/* Top-K */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Top-K
                </span>
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="1"
                  value={currentSettings.topK}
                  onChange={e => {
                    const val = parseInt(e.target.value, 10);
                    if (!isNaN(val)) handleChange('topK', Math.max(0, Math.min(100, val)));
                  }}
                  style={{
                    width: '64px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '6px',
                    background: 'rgba(0, 0, 0, 0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontWeight: 700,
                    fontSize: '0.82rem',
                    textAlign: 'right'
                  }}
                />
              </div>

              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={currentSettings.topK}
                onChange={e => handleChange('topK', parseInt(e.target.value, 10))}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-green)' }}
              />
            </div>

          </div>

          {/* Section 4: Context & Token Limits */}
          <div className="glass-card" style={{ padding: '1.1rem', display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <Layers size={16} color="var(--accent-amber)" /> Limiti di Contesto & Token
            </div>

            {/* Max Context Length */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Contesto Massimo (Token)
                </span>
                <input
                  type="number"
                  min="512"
                  max="131072"
                  step="512"
                  value={currentSettings.maxContextLength}
                  onChange={e => {
                    const val = parseInt(e.target.value, 10);
                    if (!isNaN(val)) handleChange('maxContextLength', Math.max(512, Math.min(131072, val)));
                  }}
                  style={{
                    width: '84px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '6px',
                    background: 'rgba(0, 0, 0, 0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontWeight: 700,
                    fontSize: '0.82rem',
                    textAlign: 'right'
                  }}
                />
              </div>

              <input
                type="range"
                min="512"
                max="131072"
                step="512"
                value={currentSettings.maxContextLength}
                onChange={e => handleChange('maxContextLength', parseInt(e.target.value, 10))}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-amber)' }}
              />

              <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.45rem', flexWrap: 'wrap' }}>
                {[4096, 8192, 16384, 32768, 131072].map(val => (
                  <button
                    key={val}
                    onClick={() => handleChange('maxContextLength', val)}
                    style={{
                      padding: '0.2rem 0.5rem',
                      fontSize: '0.72rem',
                      borderRadius: '5px',
                      border: '1px solid var(--border-color)',
                      background: currentSettings.maxContextLength === val ? 'var(--accent-purple)' : 'rgba(0,0,0,0.25)',
                      color: currentSettings.maxContextLength === val ? 'white' : 'var(--text-muted)',
                      cursor: 'pointer',
                      fontWeight: 600
                    }}
                  >
                    {val >= 1024 ? `${val / 1024}K` : val}
                  </button>
                ))}
              </div>
            </div>

            {/* Max Tokens Response */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  Max Tokens Risposta
                </span>
                <input
                  type="number"
                  min="16"
                  max="32768"
                  step="64"
                  value={currentSettings.maxTokens}
                  onChange={e => {
                    const val = parseInt(e.target.value, 10);
                    if (!isNaN(val)) handleChange('maxTokens', Math.max(16, Math.min(32768, val)));
                  }}
                  style={{
                    width: '84px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '6px',
                    background: 'rgba(0, 0, 0, 0.4)',
                    border: '1px solid var(--border-color)',
                    color: 'white',
                    fontWeight: 700,
                    fontSize: '0.82rem',
                    textAlign: 'right'
                  }}
                />
              </div>

              <input
                type="range"
                min="16"
                max="16384"
                step="64"
                value={currentSettings.maxTokens}
                onChange={e => handleChange('maxTokens', parseInt(e.target.value, 10))}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-purple)' }}
              />

              <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.45rem', flexWrap: 'wrap' }}>
                {[512, 1024, 2048, 4096, 8192, 16384].map(val => (
                  <button
                    key={val}
                    onClick={() => handleChange('maxTokens', val)}
                    style={{
                      padding: '0.2rem 0.5rem',
                      fontSize: '0.72rem',
                      borderRadius: '5px',
                      border: '1px solid var(--border-color)',
                      background: currentSettings.maxTokens === val ? 'var(--accent-purple)' : 'rgba(0,0,0,0.25)',
                      color: currentSettings.maxTokens === val ? 'white' : 'var(--text-muted)',
                      cursor: 'pointer',
                      fontWeight: 600
                    }}
                  >
                    {val >= 1024 ? `${val / 1024}K` : val}
                  </button>
                ))}
              </div>
            </div>

          </div>

        </div>

        {/* Panel Footer Actions */}
        <div style={{
          padding: '1.1rem 1.5rem',
          borderTop: '1px solid var(--border-color)',
          background: 'rgba(0, 0, 0, 0.15)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.65rem'
        }}>
          {saveToast && (
            <div style={{
              padding: '0.4rem 0.75rem',
              borderRadius: '6px',
              background: 'rgba(16, 185, 129, 0.2)',
              border: '1px solid rgba(16, 185, 129, 0.4)',
              color: '#34d399',
              fontSize: '0.78rem',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem'
            }}>
              <Check size={14} /> Salvate come impostazioni predefinite globali!
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button
              onClick={onResetToDefaults}
              className="btn-secondary"
              style={{ flex: 1, justifyContent: 'center', fontSize: '0.82rem', padding: '0.55rem' }}
              title="Ripristina valori predefiniti"
            >
              <RotateCcw size={14} /> Ripristina
            </button>

            <button
              onClick={handleSaveDefaults}
              className="btn-secondary"
              style={{
                flex: 1.3,
                justify: 'center',
                fontSize: '0.82rem',
                padding: '0.55rem',
                background: 'rgba(139, 92, 246, 0.15)',
                borderColor: 'rgba(139, 92, 246, 0.3)',
                color: 'var(--accent-purple)'
              }}
              title="Imposta come predefiniti per le nuove chat"
            >
              <Save size={14} /> Salva Predefiniti
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
