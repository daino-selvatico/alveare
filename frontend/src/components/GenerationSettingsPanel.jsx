import React, { useState, useRef } from 'react';
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
import { useTranslation } from '../i18n/I18nContext';
import { useFocusTrap } from '../hooks/useFocusTrap';

export default function GenerationSettingsPanel({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
  onSaveAsGlobalDefaults,
  onResetToDefaults
}) {
  const { t } = useTranslation();
  const [saveToast, setSaveToast] = useState(false);
  const containerRef = useRef(null);

  useFocusTrap({ isOpen, onClose, containerRef });

  if (!isOpen) return null;

  const currentSettings = { ...DEFAULT_SETTINGS, ...settings };

  const systemPromptPresets = [
    {
      label: t('settingsPanel.presetGeneral'),
      prompt: t('settingsPanel.presetGeneralPrompt')
    },
    {
      label: t('settingsPanel.presetSeniorDev'),
      prompt: t('settingsPanel.presetSeniorDevPrompt')
    },
    {
      label: t('settingsPanel.presetConcise'),
      prompt: t('settingsPanel.presetConcisePrompt')
    },
    {
      label: t('settingsPanel.presetReviewer'),
      prompt: t('settingsPanel.presetReviewerPrompt')
    }
  ];

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
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-panel-title"
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
        ref={containerRef}
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
              <Sliders size={20} aria-hidden="true" />
            </div>
            <div>
              <h3 id="settings-panel-title" style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                {t('settingsPanel.title')}
              </h3>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
                {t('settingsPanel.subtitle')}
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
            title={t('settingsPanel.closePanel')}
            aria-label={t('settingsPanel.closePanel')}
          >
            <X size={20} aria-hidden="true" />
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
              <label id="lbl-sys-prompt" style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <FileText size={16} color="var(--accent-purple)" aria-hidden="true" /> {t('settingsPanel.systemPrompt')}
              </label>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                {t('settingsPanel.charCount', { count: currentSettings.systemPrompt.length })}
              </span>
            </div>

            <textarea
              aria-labelledby="lbl-sys-prompt"
              value={currentSettings.systemPrompt}
              onChange={e => handleChange('systemPrompt', e.target.value)}
              rows={4}
              placeholder={t('settingsPanel.systemPromptPlaceholder')}
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
                {t('settingsPanel.quickPresets')}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {systemPromptPresets.map((preset, idx) => (
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
                    aria-label={`Imposta preset: ${preset.label}`}
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
                  <Brain size={18} aria-hidden="true" />
                </div>
                <div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>
                    {t('settingsPanel.thinkingCoT')}
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                    {currentSettings.enableThinking ? t('settingsPanel.thinkingActive') : t('settingsPanel.thinkingInactive')}
                  </div>
                </div>
              </div>

              <button
                onClick={() => handleChange('enableThinking', !currentSettings.enableThinking)}
                aria-pressed={currentSettings.enableThinking}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: currentSettings.enableThinking ? 'var(--accent-purple)' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  padding: 0
                }}
                title={t('settingsPanel.toggleThinking')}
                aria-label={t('settingsPanel.toggleThinking')}
              >
                {currentSettings.enableThinking ? <ToggleRight size={32} aria-hidden="true" /> : <ToggleLeft size={32} aria-hidden="true" />}
              </button>
            </div>
          </div>

          {/* Section 3: Sampling Parameters */}
          <div className="glass-card" style={{ padding: '1.1rem', display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <Thermometer size={16} color="var(--accent-cyan)" aria-hidden="true" /> {t('settingsPanel.samplingParams')}
            </div>

            {/* Temperature */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span id="lbl-temp" style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {t('settingsPanel.temperature')}
                </span>
                <input
                  type="number"
                  aria-labelledby="lbl-temp"
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
                aria-labelledby="lbl-temp"
                min="0"
                max="2"
                step="0.05"
                value={currentSettings.temperature}
                onChange={e => handleChange('temperature', parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-purple)' }}
              />

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                <span>{t('settingsPanel.tempPrecise')}</span>
                <span>{t('settingsPanel.tempBalanced')}</span>
                <span>{t('settingsPanel.tempCreative')}</span>
              </div>
            </div>

            {/* Top-P */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span id="lbl-topp" style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {t('settingsPanel.topP')}
                </span>
                <input
                  type="number"
                  aria-labelledby="lbl-topp"
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
                aria-labelledby="lbl-topp"
                min="0"
                max="1"
                step="0.05"
                value={currentSettings.topP}
                onChange={e => handleChange('topP', parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-cyan)' }}
              />

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                <span>{t('settingsPanel.topPFocused')}</span>
                <span>{t('settingsPanel.topPStandard')}</span>
                <span>{t('settingsPanel.topPAll')}</span>
              </div>
            </div>

            {/* Top-K */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span id="lbl-topk" style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {t('settingsPanel.topK')}
                </span>
                <input
                  type="number"
                  aria-labelledby="lbl-topk"
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
                aria-labelledby="lbl-topk"
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
              <Layers size={16} color="var(--accent-amber)" aria-hidden="true" /> {t('settingsPanel.contextLimits')}
            </div>

            {/* Max Context Length */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span id="lbl-max-ctx" style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {t('settingsPanel.maxContextLength')}
                </span>
                <input
                  type="number"
                  aria-labelledby="lbl-max-ctx"
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
                aria-labelledby="lbl-max-ctx"
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
                    aria-label={`Imposta max context a ${val >= 1024 ? `${val / 1024}K` : val}`}
                  >
                    {val >= 1024 ? `${val / 1024}K` : val}
                  </button>
                ))}
              </div>
            </div>

            {/* Max Tokens Response */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <span id="lbl-max-tok" style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {t('settingsPanel.maxTokensResponse')}
                </span>
                <input
                  type="number"
                  aria-labelledby="lbl-max-tok"
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
                aria-labelledby="lbl-max-tok"
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
                    aria-label={`Imposta max tokens a ${val >= 1024 ? `${val / 1024}K` : val}`}
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
            <div
              role="status"
              aria-live="polite"
              style={{
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
              }}
            >
              <Check size={14} aria-hidden="true" /> {t('settingsPanel.toastSavedDefaults')}
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button
              onClick={onResetToDefaults}
              className="btn-secondary"
              style={{ flex: 1, justifyContent: 'center', fontSize: '0.82rem', padding: '0.55rem' }}
              title={t('settingsPanel.resetTitle')}
              aria-label={t('settingsPanel.resetTitle')}
            >
              <RotateCcw size={14} aria-hidden="true" /> {t('settingsPanel.reset')}
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
              title={t('settingsPanel.saveDefaultsTitle')}
              aria-label={t('settingsPanel.saveDefaultsTitle')}
            >
              <Save size={14} aria-hidden="true" /> {t('settingsPanel.saveDefaults')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

