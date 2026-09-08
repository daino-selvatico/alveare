import React, { useState, useEffect, useCallback } from 'react';
import { Play, Square, RotateCw, Server, HardDrive, Settings, AlertCircle, CheckCircle, Hammer, ChevronDown, ChevronUp, Plus, Trash2, Zap, Cpu, Radio, Volume2 } from 'lucide-react';
import AddModelModal from './AddModelModal';
import { useTranslation } from '../i18n/I18nContext';

export const CANONICAL_MODEL_MAP = {
  // STT
  'whisper-base': 'openai/whisper-base',
  'whisper-large-v3-turbo': 'openai/whisper-large-v3-turbo',
  'whisper-large-v3': 'openai/whisper-large-v3',
  // TTS
  'audio8-0.1b': 'Audio8/Audio8-TTS-Preview-0.1b',
  'audio8-100m': 'Audio8/Audio8-TTS-Preview-0.1b',
  '0.1b': 'Audio8/Audio8-TTS-Preview-0.1b',
  'audio8-0.6b': 'Audio8/Audio8-TTS-Preview-0.6b',
  'audio8-600m': 'Audio8/Audio8-TTS-Preview-0.6b',
  '0.6b': 'Audio8/Audio8-TTS-Preview-0.6b',
  'kokoro': 'hexgrad/Kokoro-82M',
  'kokoro-82m': 'hexgrad/Kokoro-82M',
  'cocore': 'hexgrad/Kokoro-82M',
  'cocore-82m': 'hexgrad/Kokoro-82M',
  '82m': 'hexgrad/Kokoro-82M',
  'gemma4-e4b': 'gemma4-e4b',
  'gemma4-4b': 'gemma4-e4b',
  'e4b': 'gemma4-e4b',
  'gemma4-12b': 'gemma4',
  'g4-12b': 'gemma4',
};

export const DEFAULT_MODEL_LABELS = {
  // LLM
  'gemma4': 'Gemma 4 12B',
  'gemma4-e4b': 'Gemma 4 E4B (4B)',
  'llama': 'Llama 3.2 3B',
  // STT
  'openai/whisper-large-v3-turbo': 'Whisper Large v3 Turbo',
  'openai/whisper-large-v3': 'Whisper Large v3',
  'openai/whisper-base': 'Whisper Base',
  // TTS
  'hexgrad/Kokoro-82M': 'Kokoro 82M',
  'Audio8/Audio8-TTS-Preview-0.1b': 'Audio8 0.1B',
  'Audio8/Audio8-TTS-Preview-0.6b': 'Audio8 0.6B',
};

export const normalizeModelOption = (m) => {
  const rawId = typeof m === 'string' ? m : m.id;
  const canonicalId = CANONICAL_MODEL_MAP[rawId] || rawId;
  const label = DEFAULT_MODEL_LABELS[canonicalId] || (typeof m === 'object' && m.name) || canonicalId;
  return { id: canonicalId, label };
};

export const deduplicateModelOptions = (options) => {
  const seen = new Set();
  const result = [];
  for (const opt of options) {
    const norm = normalizeModelOption(opt);
    if (!seen.has(norm.id)) {
      seen.add(norm.id);
      result.push(norm);
    }
  }
  return result;
};

export default function ServerControl({ apiBase, status, models = [], modelsLoading = false, onRefresh }) {
  const { t } = useTranslation();
  const [loadingSlot, setLoadingSlot] = useState({}); // { llm: boolean, stt: boolean, tts: boolean, all: boolean }

  const slots = status?.slots || {};
  const llmSlot = slots.llm || {};
  const sttSlot = slots.stt || {};
  const ttsSlot = slots.tts || {};

  // Slot Model & Device selections
  const [slotModels, setSlotModels] = useState({
    llm: CANONICAL_MODEL_MAP[llmSlot.model] || llmSlot.model || 'gemma4-e4b',
    stt: CANONICAL_MODEL_MAP[sttSlot.model] || sttSlot.model || 'openai/whisper-large-v3-turbo',
    tts: CANONICAL_MODEL_MAP[ttsSlot.model] || ttsSlot.model || 'hexgrad/Kokoro-82M'
  });

  const [slotDevices, setSlotDevices] = useState({
    llm: llmSlot.device || 'gpu',
    stt: sttSlot.device || 'cpu',
    tts: ttsSlot.device || 'cpu'
  });

  // Keep state in sync with status when it arrives
  useEffect(() => {
    if (llmSlot.model) setSlotModels(prev => ({ ...prev, llm: CANONICAL_MODEL_MAP[llmSlot.model] || llmSlot.model }));
    if (sttSlot.model) setSlotModels(prev => ({ ...prev, stt: CANONICAL_MODEL_MAP[sttSlot.model] || sttSlot.model }));
    if (ttsSlot.model) setSlotModels(prev => ({ ...prev, tts: CANONICAL_MODEL_MAP[ttsSlot.model] || ttsSlot.model }));

    if (llmSlot.device) setSlotDevices(prev => ({ ...prev, llm: llmSlot.device }));
    if (sttSlot.device) setSlotDevices(prev => ({ ...prev, stt: sttSlot.device }));
    if (ttsSlot.device) setSlotDevices(prev => ({ ...prev, tts: ttsSlot.device }));
  }, [llmSlot.model, llmSlot.device, sttSlot.model, sttSlot.device, ttsSlot.model, ttsSlot.device]);

  // Server host & port settings
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
  const [buildTargetModal, setBuildTargetModal] = useState(null);
  const [buildTargetDevice, setBuildTargetDevice] = useState('npu');
  const [ttsVoice, setTtsVoice] = useState('if_sara');
  const [isAddModelOpen, setIsAddModelOpen] = useState(false);

  const fetchKernelStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/kernels/status`);
      if (res.ok) {
        const data = await res.json();
        setKernelStatus(data);
      }
    } catch (e) {
      console.error('Failed to fetch kernel status:', e);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchKernelStatus();
  }, [status?.model, fetchKernelStatus]);

  // Available model options for each slot
  const llmModelOptions = deduplicateModelOptions([
    ...models.filter(m => m.task === 'text-generation'),
    'gemma4', 'gemma4-e4b', 'llama'
  ]);

  const sttModelOptions = deduplicateModelOptions([
    ...models.filter(m => m.task === 'speech-to-text'),
    'openai/whisper-large-v3-turbo', 'openai/whisper-base'
  ]);

  const ttsModelOptions = deduplicateModelOptions([
    ...models.filter(m => m.task === 'text-to-speech'),
    'hexgrad/Kokoro-82M', 'Audio8/Audio8-TTS-Preview-0.1b', 'Audio8/Audio8-TTS-Preview-0.6b'
  ]);

  const isKokoroTts = Boolean(
    slotModels.tts?.toLowerCase().includes('kokoro') ||
    slotModels.tts?.toLowerCase().includes('cocore') ||
    slotModels.tts?.includes('82m')
  );

  useEffect(() => {
    if (isKokoroTts && slotDevices.tts === 'npu') {
      setSlotDevices(prev => ({ ...prev, tts: 'cpu' }));
    }
  }, [isKokoroTts, slotDevices.tts]);

  // Slot start & stop handlers
  const handleStartSlot = async (slotId) => {
    const model = slotModels[slotId];
    const device = slotDevices[slotId];
    setLoadingSlot(prev => ({ ...prev, [slotId]: true }));
    try {
      await fetch(`${apiBase}/api/control/slot/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot: slotId, model, device, host, port })
      });
      setTimeout(() => {
        if (onRefresh) onRefresh();
        fetchKernelStatus();
      }, 1000);
    } catch (e) {
      alert(t('serverControl.startError', { error: e.message || e }));
    } finally {
      setLoadingSlot(prev => ({ ...prev, [slotId]: false }));
    }
  };

  const handleStopSlot = async (slotId) => {
    setLoadingSlot(prev => ({ ...prev, [slotId]: true }));
    try {
      await fetch(`${apiBase}/api/control/slot/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot: slotId })
      });
      setTimeout(() => {
        if (onRefresh) onRefresh();
        fetchKernelStatus();
      }, 800);
    } catch (e) {
      alert(t('serverControl.stopError', { error: e.message || e }));
    } finally {
      setLoadingSlot(prev => ({ ...prev, [slotId]: false }));
    }
  };

  // Master Tri-Hardware Live Start & Stop
  const handleMasterStart = async () => {
    setLoadingSlot(prev => ({ ...prev, all: true }));
    try {
      await fetch(`${apiBase}/api/live/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          llm_model: slotModels.llm || 'gemma4-e4b',
          llm_device: slotDevices.llm || 'gpu',
          stt_model: slotModels.stt || 'openai/whisper-large-v3-turbo',
          stt_device: slotDevices.stt || 'cpu',
          tts_model: slotModels.tts || 'hexgrad/Kokoro-82M',
          tts_device: isKokoroTts && slotDevices.tts === 'npu' ? 'cpu' : (slotDevices.tts || (isKokoroTts ? 'cpu' : 'npu'))
        })
      });
      setTimeout(() => {
        if (onRefresh) onRefresh();
        fetchKernelStatus();
      }, 1200);
    } catch (e) {
      alert(t('serverControl.startError', { error: e.message || e }));
    } finally {
      setLoadingSlot(prev => ({ ...prev, all: false }));
    }
  };

  const handleMasterStop = async () => {
    setLoadingSlot(prev => ({ ...prev, all: true }));
    try {
      await fetch(`${apiBase}/api/live/stop`, { method: 'POST' });
      setTimeout(() => {
        if (onRefresh) onRefresh();
        fetchKernelStatus();
      }, 800);
    } catch (e) {
      alert(t('serverControl.stopError', { error: e.message || e }));
    } finally {
      setLoadingSlot(prev => ({ ...prev, all: false }));
    }
  };

  const handleDeleteModel = async (modelId) => {
    if (status?.is_running && (status?.model === modelId || llmSlot.model === modelId)) {
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
        if (onRefresh) onRefresh();
        fetchKernelStatus();
      } else {
        const err = await res.json();
        alert(t('serverControl.deleteError', { error: err.detail || 'Failed' }));
      }
    } catch (e) {
      alert(t('serverControl.deleteError', { error: e }));
    }
  };

  const handleBuildKernels = async (modelId, deviceOverride) => {
    const device = deviceOverride || buildTargetDevice || slotDevices.llm || 'npu';
    if (!confirm(t('serverControl.buildKernelConfirm', { modelId, device: device.toUpperCase() }))) {
      return;
    }
    setBuildingModel(modelId);
    setBuildTargetModal(null);
    try {
      await fetch(`${apiBase}/api/control/build-kernels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelId,
          device: device,
          no_gemm: noGemm,
          max_batch: maxBatch
        })
      });
      alert(t('serverControl.buildKernelStarted', { modelId, device: device.toUpperCase() }));
      setTimeout(fetchKernelStatus, 3000);
    } catch (e) {
      alert(t('serverControl.buildKernelError', { error: e }));
    } finally {
      setBuildingModel(null);
    }
  };

  const activeSlotsCount = [llmSlot, sttSlot, ttsSlot].filter(s => s.status === 'running' || s.is_loaded).length;

  return (
    <div
      role="region"
      aria-label={t('serverControl.engineController')}
      className="responsive-page-container"
      style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', width: '100%' }}
    >
      {/* 1. MASTER TRI-HARDWARE CONTROLLER BANNER */}
      <div className="glass-card" style={{ padding: 'clamp(1.25rem, 3vw, 1.75rem)', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
              <Server size={26} color="var(--accent-purple)" aria-hidden="true" />
              <h2 style={{ fontSize: 'clamp(1.2rem, 3vw, 1.45rem)', fontWeight: 800 }}>{t('serverControl.engineController')}</h2>
              
              {activeSlotsCount === 3 ? (
                <span className="badge badge-success">
                  <CheckCircle size={14} aria-hidden="true" /> {t('serverControl.allSlotsActive')} (3/3)
                </span>
              ) : activeSlotsCount > 0 ? (
                <span className="badge badge-warning">
                  <RotateCw size={14} className="pulse-icon" aria-hidden="true" /> {activeSlotsCount}/3 Slot Attivi
                </span>
              ) : (
                <span className="badge badge-danger">
                  <AlertCircle size={14} aria-hidden="true" /> {t('serverControl.statusStopped')}
                </span>
              )}
            </div>
            
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', maxWidth: '650px', lineHeight: '1.5' }}>
              {t('serverControl.triHardwareDesc')}
            </p>
          </div>

          {/* Master 1-Click Action Buttons */}
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', width: 'auto' }}>
            <button
              className="btn-primary mobile-touch-target"
              onClick={handleMasterStart}
              disabled={loadingSlot.all}
              style={{
                background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)',
                padding: '0.75rem 1.4rem',
                fontSize: '0.92rem',
                fontWeight: 700
              }}
              aria-label={t('serverControl.triHardwareMaster')}
            >
              <Play size={18} className={loadingSlot.all ? 'pulse-icon' : ''} aria-hidden="true" />
              {loadingSlot.all ? t('serverControl.starting') : t('serverControl.triHardwareMaster')}
            </button>

            <button
              className="btn-danger mobile-touch-target"
              onClick={handleMasterStop}
              disabled={loadingSlot.all || activeSlotsCount === 0}
              style={{ padding: '0.75rem 1.25rem', fontSize: '0.92rem' }}
              aria-label={t('serverControl.stopAllSlots')}
            >
              <Square size={18} aria-hidden="true" />
              {t('serverControl.stopAllSlots')}
            </button>
          </div>
        </div>
      </div>

      {/* 2. DEDICATED 3-SLOT CARDS (LLM, STT, TTS) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))',
        gap: '1.25rem',
        width: '100%'
      }}>
        {/* SLOT 1: LLM Engine (Text Generation) */}
        <div className="glass-card" style={{
          padding: '1.4rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          borderLeft: '4px solid #c084fc',
          background: llmSlot.status === 'running' ? 'rgba(168, 85, 247, 0.06)' : 'var(--bg-card)'
        }}>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#c084fc', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Slot 1 • LLM Engine
              </span>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginTop: '0.2rem', color: 'var(--text-main)' }}>
                {t('serverControl.slotLlmTitle')}
              </h3>
            </div>
            <div>
              {llmSlot.status === 'running' && llmSlot.is_loaded ? (
                <span className="badge badge-success"><CheckCircle size={12} /> {t('serverControl.slotStatusActive')}</span>
              ) : llmSlot.status === 'starting' ? (
                <span className="badge badge-warning"><RotateCw size={12} className="pulse-icon" /> {t('serverControl.slotStatusStarting')}</span>
              ) : llmSlot.status === 'error' ? (
                <span className="badge badge-danger"><AlertCircle size={12} /> {t('serverControl.slotStatusError')}</span>
              ) : (
                <span className="badge badge-secondary">{t('serverControl.slotStatusStopped')}</span>
              )}
            </div>
          </div>

          {/* Model Selector */}
          <div>
            <label htmlFor="slot-llm-model" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              {t('serverControl.slotModelLabel')}
            </label>
            <select
              id="slot-llm-model"
              value={slotModels.llm}
              onChange={e => setSlotModels(prev => ({ ...prev, llm: e.target.value }))}
              style={{
                width: '100%',
                padding: '0.65rem 0.75rem',
                borderRadius: '8px',
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-main)',
                fontSize: '0.88rem',
                fontFamily: 'var(--font-mono)'
              }}
            >
              {llmModelOptions.map(opt => (
                <option key={opt.id} value={opt.id} style={{ background: '#131b2e' }}>{opt.label}</option>
              ))}
            </select>
          </div>

          {/* Hardware Device Selector (GPU, NPU, CPU) */}
          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              {t('serverControl.slotDeviceLabel')}
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.4rem' }}>
              {[
                { id: 'gpu', label: '🎮 GPU', name: 'Radeon 890M', color: '#c084fc' },
                { id: 'npu', label: '⚡ NPU', name: 'XDNA2', color: '#34d399' },
                { id: 'cpu', label: '🖥️ CPU', name: 'Host AVX2', color: '#60a5fa' }
              ].map(dev => (
                <button
                  key={dev.id}
                  type="button"
                  className="mobile-touch-target"
                  onClick={() => setSlotDevices(prev => ({ ...prev, llm: dev.id }))}
                  style={{
                    padding: '0.45rem 0.3rem',
                    borderRadius: '8px',
                    border: slotDevices.llm === dev.id ? `2px solid ${dev.color}` : '1px solid var(--border-color)',
                    background: slotDevices.llm === dev.id ? `${dev.color}22` : 'rgba(0,0,0,0.25)',
                    color: slotDevices.llm === dev.id ? dev.color : 'var(--text-muted)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.1rem',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <span>{dev.label}</span>
                  <span style={{ fontSize: '0.65rem', opacity: 0.8 }}>{dev.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Real-time telemetry badges */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Dispositivo:</span>
            <span style={{ fontWeight: 700, color: slotDevices.llm === 'gpu' ? '#c084fc' : slotDevices.llm === 'npu' ? '#34d399' : '#60a5fa' }}>
              {slotDevices.llm === 'gpu' ? '🎮 GPU (Vulkan 1.4 Native)' : slotDevices.llm === 'npu' ? '⚡ NPU (Ryzen AI)' : '🖥️ CPU Multi-Thread'}
            </span>
          </div>

          {llmSlot.tok_per_sec > 0 && (
            <div style={{ fontSize: '0.82rem', padding: '0.4rem 0.75rem', borderRadius: '6px', background: 'rgba(6, 182, 212, 0.15)', border: '1px solid rgba(6, 182, 212, 0.3)', color: 'var(--accent-cyan)', display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
              <span>{t('serverControl.measuredSpeed')}</span>
              <span>⚡ {llmSlot.tok_per_sec} tok/s</span>
            </div>
          )}

          {/* Slot Action Button */}
          <div style={{ marginTop: 'auto', paddingTop: '0.5rem' }}>
            {llmSlot.status === 'running' || llmSlot.status === 'starting' ? (
              <button
                className="btn-danger mobile-touch-target"
                onClick={() => handleStopSlot('llm')}
                disabled={loadingSlot.llm}
                style={{ width: '100%', justifyContent: 'center', minHeight: '46px' }}
                aria-label={t('serverControl.stopSlot')}
              >
                <Square size={16} /> {t('serverControl.stopSlot')}
              </button>
            ) : (
              <button
                className="btn-primary mobile-touch-target"
                onClick={() => handleStartSlot('llm')}
                disabled={loadingSlot.llm}
                style={{ width: '100%', justifyContent: 'center', minHeight: '46px', background: 'linear-gradient(135deg, #a855f7 0%, #06b6d4 100%)' }}
                aria-label={t('serverControl.startSlot')}
              >
                <Play size={16} className={loadingSlot.llm ? 'pulse-icon' : ''} /> {t('serverControl.startSlot')}
              </button>
            )}
          </div>
        </div>

        {/* SLOT 2: STT Engine (Speech-to-Text) */}
        <div className="glass-card" style={{
          padding: '1.4rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          borderLeft: '4px solid #60a5fa',
          background: sttSlot.status === 'running' ? 'rgba(96, 165, 250, 0.06)' : 'var(--bg-card)'
        }}>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Slot 2 • STT Engine
              </span>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginTop: '0.2rem', color: 'var(--text-main)' }}>
                {t('serverControl.slotSttTitle')}
              </h3>
            </div>
            <div>
              {sttSlot.status === 'running' && sttSlot.is_loaded ? (
                <span className="badge badge-success"><CheckCircle size={12} /> {t('serverControl.slotStatusActive')}</span>
              ) : sttSlot.status === 'starting' ? (
                <span className="badge badge-warning"><RotateCw size={12} className="pulse-icon" /> {t('serverControl.slotStatusStarting')}</span>
              ) : sttSlot.status === 'error' ? (
                <span className="badge badge-danger"><AlertCircle size={12} /> {t('serverControl.slotStatusError')}</span>
              ) : (
                <span className="badge badge-secondary">{t('serverControl.slotStatusStopped')}</span>
              )}
            </div>
          </div>

          {/* Model Selector */}
          <div>
            <label htmlFor="slot-stt-model" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              {t('serverControl.slotModelLabel')}
            </label>
            <select
              id="slot-stt-model"
              value={slotModels.stt}
              onChange={e => setSlotModels(prev => ({ ...prev, stt: e.target.value }))}
              style={{
                width: '100%',
                padding: '0.65rem 0.75rem',
                borderRadius: '8px',
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-main)',
                fontSize: '0.88rem',
                fontFamily: 'var(--font-mono)'
              }}
            >
              {sttModelOptions.map(opt => (
                <option key={opt.id} value={opt.id} style={{ background: '#131b2e' }}>{opt.label}</option>
              ))}
            </select>
          </div>

          {/* Hardware Device Selector (CPU default, NPU, GPU) */}
          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              {t('serverControl.slotDeviceLabel')}
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.4rem' }}>
              {[
                { id: 'cpu', label: '🖥️ CPU', name: 'Multi-Core', color: '#60a5fa' },
                { id: 'npu', label: '⚡ NPU', name: 'XDNA2', color: '#34d399' },
                { id: 'gpu', label: '🎮 GPU', name: 'Radeon 890M', color: '#c084fc' }
              ].map(dev => (
                <button
                  key={dev.id}
                  type="button"
                  className="mobile-touch-target"
                  onClick={() => setSlotDevices(prev => ({ ...prev, stt: dev.id }))}
                  style={{
                    padding: '0.45rem 0.3rem',
                    borderRadius: '8px',
                    border: slotDevices.stt === dev.id ? `2px solid ${dev.color}` : '1px solid var(--border-color)',
                    background: slotDevices.stt === dev.id ? `${dev.color}22` : 'rgba(0,0,0,0.25)',
                    color: slotDevices.stt === dev.id ? dev.color : 'var(--text-muted)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.1rem',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <span>{dev.label}</span>
                  <span style={{ fontSize: '0.65rem', opacity: 0.8 }}>{dev.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Real-time telemetry badges */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Dispositivo:</span>
            <span style={{ fontWeight: 700, color: slotDevices.stt === 'cpu' ? '#60a5fa' : slotDevices.stt === 'npu' ? '#34d399' : '#c084fc' }}>
              {slotDevices.stt === 'cpu' ? '🖥️ CPU Multi-Thread (Ottimale)' : slotDevices.stt === 'npu' ? '⚡ NPU (Ryzen AI)' : '🎮 GPU (Vulkan 1.4)'}
            </span>
          </div>

          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: '0.35rem 0.5rem' }}>
            Trascrizione audio e VAD in streaming continuo senza interferire con l'inferenza LLM.
          </div>

          {/* Slot Action Button */}
          <div style={{ marginTop: 'auto', paddingTop: '0.5rem' }}>
            {sttSlot.status === 'running' || sttSlot.status === 'starting' ? (
              <button
                className="btn-danger mobile-touch-target"
                onClick={() => handleStopSlot('stt')}
                disabled={loadingSlot.stt}
                style={{ width: '100%', justifyContent: 'center', minHeight: '46px' }}
                aria-label={t('serverControl.stopSlot')}
              >
                <Square size={16} /> {t('serverControl.stopSlot')}
              </button>
            ) : (
              <button
                className="btn-primary mobile-touch-target"
                onClick={() => handleStartSlot('stt')}
                disabled={loadingSlot.stt}
                style={{ width: '100%', justifyContent: 'center', minHeight: '46px', background: 'linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)' }}
                aria-label={t('serverControl.startSlot')}
              >
                <Play size={16} className={loadingSlot.stt ? 'pulse-icon' : ''} /> {t('serverControl.startSlot')}
              </button>
            )}
          </div>
        </div>

        {/* SLOT 3: TTS Engine (Text-to-Speech) */}
        <div className="glass-card" style={{
          padding: '1.4rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          borderLeft: '4px solid #34d399',
          background: ttsSlot.status === 'running' ? 'rgba(52, 211, 153, 0.06)' : 'var(--bg-card)'
        }}>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#34d399', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Slot 3 • TTS Engine
              </span>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginTop: '0.2rem', color: 'var(--text-main)' }}>
                {t('serverControl.slotTtsTitle')}
              </h3>
            </div>
            <div>
              {ttsSlot.status === 'running' && ttsSlot.is_loaded ? (
                <span className="badge badge-success"><CheckCircle size={12} /> {t('serverControl.slotStatusActive')}</span>
              ) : ttsSlot.status === 'starting' ? (
                <span className="badge badge-warning"><RotateCw size={12} className="pulse-icon" /> {t('serverControl.slotStatusStarting')}</span>
              ) : ttsSlot.status === 'error' ? (
                <span className="badge badge-danger"><AlertCircle size={12} /> {t('serverControl.slotStatusError')}</span>
              ) : (
                <span className="badge badge-secondary">{t('serverControl.slotStatusStopped')}</span>
              )}
            </div>
          </div>

          {/* Model Selector */}
          <div>
            <label htmlFor="slot-tts-model" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              {t('serverControl.slotModelLabel')}
            </label>
            <select
              id="slot-tts-model"
              value={slotModels.tts}
              onChange={e => setSlotModels(prev => ({ ...prev, tts: e.target.value }))}
              style={{
                width: '100%',
                padding: '0.65rem 0.75rem',
                borderRadius: '8px',
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-main)',
                fontSize: '0.88rem',
                fontFamily: 'var(--font-mono)'
              }}
            >
              {ttsModelOptions.map(opt => (
                <option key={opt.id} value={opt.id} style={{ background: '#131b2e' }}>{opt.label}</option>
              ))}
            </select>
          </div>

          {/* Hardware Device Selector (NPU default, CPU, GPU) */}
          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
              {t('serverControl.slotDeviceLabel')}
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.4rem' }}>
              {[
                { id: 'npu', label: '⚡ NPU', name: 'XDNA2', color: '#34d399', disabled: isKokoroTts },
                { id: 'cpu', label: '🖥️ CPU', name: isKokoroTts ? 'Zen 5 AVX-512' : 'Multi-Core', color: '#60a5fa' },
                { id: 'gpu', label: '🎮 GPU', name: isKokoroTts ? 'CUDA / Direct' : 'Radeon 890M', color: '#c084fc' }
              ].map(dev => (
                <button
                  key={dev.id}
                  type="button"
                  disabled={dev.disabled}
                  className="mobile-touch-target"
                  onClick={() => !dev.disabled && setSlotDevices(prev => ({ ...prev, tts: dev.id }))}
                  title={dev.disabled ? 'NPU non supportata per architettura StyleTTS2 (Richiede CPU Zen 5 o GPU CUDA)' : ''}
                  style={{
                    padding: '0.45rem 0.3rem',
                    borderRadius: '8px',
                    border: slotDevices.tts === dev.id ? `2px solid ${dev.color}` : '1px solid var(--border-color)',
                    background: dev.disabled ? 'rgba(0,0,0,0.4)' : (slotDevices.tts === dev.id ? `${dev.color}22` : 'rgba(0,0,0,0.25)'),
                    color: dev.disabled ? '#475569' : (slotDevices.tts === dev.id ? dev.color : 'var(--text-muted)'),
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    cursor: dev.disabled ? 'not-allowed' : 'pointer',
                    opacity: dev.disabled ? 0.4 : 1,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.1rem',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <span>{dev.label}</span>
                  <span style={{ fontSize: '0.65rem', opacity: 0.8 }}>{dev.disabled ? 'Non supp.' : dev.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Real-time telemetry badges */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Dispositivo:</span>
            <span style={{ fontWeight: 700, color: slotDevices.tts === 'npu' ? '#34d399' : slotDevices.tts === 'cpu' ? '#60a5fa' : '#c084fc' }}>
              {isKokoroTts
                ? (slotDevices.tts === 'gpu' ? '🎮 GPU (CUDA / Direct)' : '🖥️ CPU Multi-Thread Zen 5 (Consigliato RTF ~0.10x)')
                : (slotDevices.tts === 'npu' ? '⚡ NPU Ryzen AI (Hardware Native)' : slotDevices.tts === 'cpu' ? '🖥️ CPU Multi-Thread' : '🎮 GPU (Vulkan 1.4)')}
            </span>
          </div>

          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: '0.35rem 0.5rem' }}>
            {isKokoroTts
              ? 'Sintesi vocale Kokoro-82M neurale non-autoregressiva ultraveloce (RTF ~0.10x su CPU Zen 5) con prosodia italiana naturale.'
              : 'Sintesi vocale Audio8 a bassissima latenza con voice cloning zero-shot ed espressività italiana.'}
          </div>

          {/* TTS Parameters Config Box */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', padding: '0.6rem 0.75rem', background: 'rgba(0,0,0,0.25)', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.8rem' }}>
            <span style={{ fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.75rem' }}>
              {isKokoroTts ? 'PARAMETRI KOKORO TTS' : 'PARAMETRI AUDIO8 TTS'}
            </span>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Voce Default:</span>
              <select
                value={ttsVoice}
                onChange={e => setTtsVoice(e.target.value)}
                style={{ padding: '0.25rem 0.5rem', borderRadius: '6px', background: 'rgba(0,0,0,0.4)', color: 'var(--text-main)', border: '1px solid var(--border-color)', fontSize: '0.78rem' }}
              >
                {isKokoroTts ? (
                  <>
                    <option value="if_sara">if_sara (Italiana Femminile)</option>
                    <option value="im_nicola">im_nicola (Italiano Maschile)</option>
                    <option value="af_heart">af_heart (Melodica HeartMuLa)</option>
                  </>
                ) : (
                  <>
                    <option value="valeria">valeria (Ufficiale Alveare)</option>
                    <option value="daino">daino (Italiano Maschile)</option>
                    <option value="bolt">bolt (Conciso)</option>
                  </>
                )}
              </select>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Campionamento:</span>
              <span style={{ fontWeight: 600, color: '#34d399' }}>
                {isKokoroTts ? '24000 Hz (StyleTTS2)' : '44100 Hz (DualAR Mamba)'}
              </span>
            </div>
          </div>

          {/* Slot Action Button */}
          <div style={{ marginTop: 'auto', paddingTop: '0.5rem' }}>
            {ttsSlot.status === 'running' || ttsSlot.status === 'starting' ? (
              <button
                className="btn-danger mobile-touch-target"
                onClick={() => handleStopSlot('tts')}
                disabled={loadingSlot.tts}
                style={{ width: '100%', justifyContent: 'center', minHeight: '46px' }}
                aria-label={t('serverControl.stopSlot')}
              >
                <Square size={16} /> {t('serverControl.stopSlot')}
              </button>
            ) : (
              <button
                className="btn-primary mobile-touch-target"
                onClick={() => handleStartSlot('tts')}
                disabled={loadingSlot.tts}
                style={{ width: '100%', justifyContent: 'center', minHeight: '46px', background: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)' }}
                aria-label={t('serverControl.startSlot')}
              >
                <Play size={16} className={loadingSlot.tts ? 'pulse-icon' : ''} /> {t('serverControl.startSlot')}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* 3. INSTALLED QUANTIZED MODELS GALLERY */}
      <div style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-main)' }}>
            <HardDrive size={22} color="var(--accent-cyan)" /> {t('serverControl.availableModels')}
          </h3>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <button
              onClick={() => setIsAddModelOpen(true)}
              className="btn-primary mobile-touch-target"
              style={{
                padding: '0.55rem 1.1rem',
                fontSize: '0.88rem',
                fontWeight: 700,
                background: 'var(--accent-purple)'
              }}
              aria-label={t('serverControl.addModel')}
            >
              <Plus size={16} /> {t('serverControl.addModel')}
            </button>
          </div>
        </div>

        {/* Loading Skeletons */}
        {modelsLoading ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 300px), 1fr))', gap: '1.25rem' }}>
            {[1, 2].map(n => (
              <div
                key={n}
                className="glass-card pulse-icon"
                style={{ height: '180px', borderRadius: '12px', opacity: 0.6 }}
              />
            ))}
          </div>
        ) : models.length === 0 ? (
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
              justifyContent: 'center'
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
              className="btn-primary mobile-touch-target"
              style={{ padding: '0.75rem 1.5rem', fontSize: '0.95rem' }}
              aria-label={t('serverControl.downloadOrImportFirst')}
            >
              <Plus size={18} /> {t('serverControl.downloadOrImportFirst')}
            </button>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))', gap: '1.25rem' }}>
            {models.map(m => {
              const isLlmServed = llmSlot.status === 'running' && (llmSlot.model === m.id || CANONICAL_MODEL_MAP[llmSlot.model] === m.id || llmSlot.model === CANONICAL_MODEL_MAP[m.id]);
              const isSttServed = sttSlot.status === 'running' && (sttSlot.model === m.id || CANONICAL_MODEL_MAP[sttSlot.model] === m.id || sttSlot.model === CANONICAL_MODEL_MAP[m.id]);
              const isTtsServed = ttsSlot.status === 'running' && (ttsSlot.model === m.id || CANONICAL_MODEL_MAP[ttsSlot.model] === m.id || ttsSlot.model === CANONICAL_MODEL_MAP[m.id]);
              const isServed = isLlmServed || isSttServed || isTtsServed;

              const mKernel = kernelStatus?.[m.id] || (kernelStatus?.manifest_model_type === m.arch ? kernelStatus : null);
              const isKernelMatching = mKernel?.manifest_exists;

              return (
                <div
                  key={m.id}
                  className="glass-card"
                  style={{
                    padding: '1.35rem',
                    border: isServed ? '2px solid #10b981' : '1px solid var(--border-color)',
                    background: isServed ? 'rgba(16, 185, 129, 0.08)' : 'rgba(23, 32, 54, 0.6)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '1rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                    <div>
                      <h4 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-main)' }}>
                        {m.id}
                      </h4>
                      <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                        {t('serverControl.arch', { arch: m.arch, size: m.size_mb })}
                      </span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.3rem' }}>
                      {isServed && <span className="badge badge-success"><CheckCircle size={12} /> {t('serverControl.inExecution')}</span>}
                      <span className="badge badge-secondary" style={{ fontSize: '0.7rem' }}>
                        {m.task === 'speech-to-text' ? 'STT' : m.task === 'text-to-speech' ? 'TTS' : 'LLM'}
                      </span>
                    </div>
                  </div>

                  {/* Multi-Hardware Acceleration Status */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', padding: '0.65rem 0.8rem', borderRadius: '8px', background: 'rgba(0,0,0,0.35)', border: '1px solid var(--border-color)', fontSize: '0.8rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.72rem', letterSpacing: '0.04em' }}>
                        {t('serverControl.hardwareKernels')}
                      </span>
                    </div>

                    {/* NPU Backend */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ color: '#34d399', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        ⚡ NPU:
                      </span>
                      <span style={{ fontWeight: 600, color: (mKernel?.manifest_exists || isKernelMatching) ? '#34d399' : (mKernel?.is_kokoro ? 'var(--text-muted)' : '#fbbf24'), fontSize: '0.78rem' }}>
                        {mKernel?.backends?.npu?.info || ((mKernel?.manifest_exists || isKernelMatching) ? t('serverControl.compiledCount', { count: mKernel?.kernels_count || 0 }) : (mKernel?.is_kokoro ? 'Non applicabile (StyleTTS2)' : t('serverControl.notCompiled')))}
                      </span>
                    </div>

                    {/* GPU Backend */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ color: '#c084fc', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        🎮 GPU:
                      </span>
                      <span style={{ fontWeight: 600, color: '#c084fc', fontSize: '0.78rem' }}>
                        {mKernel?.backends?.gpu?.info || (mKernel?.is_tts ? 'PyTorch CUDA / Direct' : 'SPIR-V Shaders Vulkan 1.4')}
                      </span>
                    </div>

                    {/* CPU Backend */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ color: '#60a5fa', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        🖥️ CPU:
                      </span>
                      <span style={{ fontWeight: 600, color: '#60a5fa', fontSize: '0.78rem' }}>
                        {mKernel?.backends?.cpu?.info || 'AVX-512 / OpenMP (Nativo)'}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div style={{ display: 'flex', gap: '0.6rem', marginTop: 'auto', paddingTop: '0.5rem', flexWrap: 'wrap' }}>
                    <button
                      className="btn-secondary mobile-touch-target"
                      style={{ flex: 1, padding: '0.5rem 0.75rem', fontSize: '0.85rem', color: '#c084fc', borderColor: 'rgba(139,92,246,0.4)', justifyContent: 'center' }}
                      onClick={() => setBuildTargetModal({ modelId: m.id })}
                      disabled={buildingModel === m.id}
                      title={t('serverControl.compileKernelTitle')}
                      aria-label={`Compila kernel per ${m.id}`}
                    >
                      <Hammer size={14} /> {buildingModel === m.id ? t('serverControl.compiling') : t('serverControl.compileKernel')}
                    </button>

                    <button
                      className="btn-danger mobile-touch-target"
                      style={{ padding: '0.5rem 0.85rem', fontSize: '0.85rem' }}
                      onClick={() => handleDeleteModel(m.id)}
                      disabled={isServed}
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

      {/* 4. SERVER CONFIG & ADVANCED OPTIONS */}
      <div className="glass-card" style={{ padding: 'clamp(1rem, 3vw, 1.5rem)', width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Settings size={20} color="var(--accent-cyan)" /> {t('serverControl.serverConfig')}
          </h3>

          <button
            className="btn-secondary mobile-touch-target"
            onClick={() => setShowBuildOptions(!showBuildOptions)}
            style={{ padding: '0.4rem 0.75rem', fontSize: '0.85rem' }}
            aria-expanded={showBuildOptions}
          >
            {showBuildOptions ? <ChevronUp size={16} /> : <ChevronDown size={16} />} {t('serverControl.advancedCompilationOptions')}
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 220px), 1fr))', gap: '1.25rem' }}>
          <div>
            <label htmlFor="server-host" style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              {t('serverControl.hostAddress')}
            </label>
            <input
              id="server-host"
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
            <label htmlFor="server-port" style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              {t('serverControl.httpPort')}
            </label>
            <input
              id="server-port"
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
          if (onRefresh) onRefresh();
          setSlotModels(prev => ({ ...prev, llm: alias }));
        }}
      />

      {/* Build Target Hardware Modal */}
      {buildTargetModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="build-modal-title"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.75)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '1rem'
          }}
        >
          <div
            className="glass-card"
            style={{
              maxWidth: '540px',
              width: '100%',
              padding: '1.75rem',
              borderRadius: '16px',
              border: '1px solid rgba(139,92,246,0.4)',
              background: '#0f172a',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.25rem',
              boxShadow: '0 20px 25px -5px rgba(0,0,0,0.6)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Hammer size={22} color="var(--accent-purple)" />
                <h3 id="build-modal-title" style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0 }}>
                  {t('serverControl.compileKernel')}
                </h3>
              </div>
              <button
                className="btn-secondary"
                onClick={() => setBuildTargetModal(null)}
                style={{ padding: '0.35rem 0.6rem' }}
              >
                <X size={16} />
              </button>
            </div>

            <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)' }}>
              Modello: <strong style={{ color: 'var(--text-main)' }}>{buildTargetModal.modelId}</strong>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                {t('serverControl.targetHardware')}
              </span>

              {/* Target 1: NPU */}
              <label
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  borderRadius: '10px',
                  border: buildTargetDevice === 'npu' ? '2px solid #34d399' : '1px solid var(--border-color)',
                  background: buildTargetDevice === 'npu' ? 'rgba(52, 211, 153, 0.1)' : 'rgba(0,0,0,0.25)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <input
                  type="radio"
                  name="buildTarget"
                  checked={buildTargetDevice === 'npu'}
                  onChange={() => setBuildTargetDevice('npu')}
                  style={{ marginTop: '0.2rem', accentColor: '#34d399' }}
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                  <span style={{ fontWeight: 700, color: '#34d399' }}>
                    {t('serverControl.targetNpuDesc')}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                    Compila micro-kernel AIE2 (.xclbin) per GEMV Q4 / prefill GEMM. Richiede compilazione XDNA2.
                  </span>
                </div>
              </label>

              {/* Target 2: GPU */}
              <label
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  borderRadius: '10px',
                  border: buildTargetDevice === 'gpu' ? '2px solid #c084fc' : '1px solid var(--border-color)',
                  background: buildTargetDevice === 'gpu' ? 'rgba(192, 132, 252, 0.1)' : 'rgba(0,0,0,0.25)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <input
                  type="radio"
                  name="buildTarget"
                  checked={buildTargetDevice === 'gpu'}
                  onChange={() => setBuildTargetDevice('gpu')}
                  style={{ marginTop: '0.2rem', accentColor: '#c084fc' }}
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                  <span style={{ fontWeight: 700, color: '#c084fc' }}>
                    {t('serverControl.targetGpuDesc')}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                    Compila compute shaders SPIR-V (Vulkan 1.3/1.4) con glslc / glslangValidator.
                  </span>
                </div>
              </label>

              {/* Target 3: CPU */}
              <label
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  borderRadius: '10px',
                  border: buildTargetDevice === 'cpu' ? '2px solid #60a5fa' : '1px solid var(--border-color)',
                  background: buildTargetDevice === 'cpu' ? 'rgba(96, 165, 250, 0.1)' : 'rgba(0,0,0,0.25)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <input
                  type="radio"
                  name="buildTarget"
                  checked={buildTargetDevice === 'cpu'}
                  onChange={() => setBuildTargetDevice('cpu')}
                  style={{ marginTop: '0.2rem', accentColor: '#60a5fa' }}
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                  <span style={{ fontWeight: 700, color: '#60a5fa' }}>
                    {t('serverControl.targetCpuDesc')}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.3' }}>
                    Verifica e compila il motore C++ nativo AVX-512 / OpenMP per CPU Zen 5.
                  </span>
                </div>
              </label>
            </div>

            {/* Modal Actions */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
              <button
                className="btn-secondary"
                onClick={() => setBuildTargetModal(null)}
              >
                Annulla
              </button>
              <button
                className="btn-primary"
                onClick={() => handleBuildKernels(buildTargetModal.modelId, buildTargetDevice)}
                style={{ background: 'linear-gradient(135deg, #a855f7 0%, #06b6d4 100%)' }}
              >
                Avvia Compilazione
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
