import React, { useState, useMemo } from 'react';
import { FileText, X, Check, Sliders, Sparkles, BookOpen } from 'lucide-react';
import { formatPdfDocumentText, estimateTokens, parsePageRange } from '../utils/documentParser';

export default function DocumentConfigModal({
  attachment,
  onSave,
  onClose
}) {
  if (!attachment) return null;

  const isPdf = attachment.pdfData && attachment.pdfData.pages && attachment.pdfData.pages.length > 0;
  const totalPages = isPdf ? attachment.pdfData.pages.length : 1;

  const [pageSelection, setPageSelection] = useState(attachment.pageSelection || 'all');
  const [customRange, setCustomRange] = useState(attachment.customRange || '1-3');
  const [cleanText, setCleanText] = useState(attachment.cleanText !== false);

  // Compute live preview
  const previewText = useMemo(() => {
    if (isPdf) {
      return formatPdfDocumentText(attachment.pdfData.pages, {
        pageSelection,
        customRange,
        cleanText
      });
    }
    return attachment.textData || '';
  }, [isPdf, attachment, pageSelection, customRange, cleanText]);

  const tokenCount = useMemo(() => estimateTokens(previewText), [previewText]);

  const includedPages = useMemo(() => {
    if (!isPdf) return [1];
    if (pageSelection === 'all') return Array.from({ length: totalPages }, (_, i) => i + 1);
    if (pageSelection === 'first3') return Array.from({ length: Math.min(3, totalPages) }, (_, i) => i + 1);
    if (pageSelection === 'custom') return parsePageRange(customRange, totalPages);
    return [];
  }, [isPdf, pageSelection, customRange, totalPages]);

  const handleApply = () => {
    onSave({
      id: attachment.id,
      textData: previewText,
      pageSelection,
      customRange,
      cleanText
    });
    onClose();
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.75)',
        backdropFilter: 'blur(8px)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem'
      }}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="doc-config-title"
    >
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '16px',
          width: '100%',
          maxWidth: '680px',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
          overflow: 'hidden'
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(255, 255, 255, 0.02)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              background: 'rgba(16, 185, 129, 0.15)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <FileText size={20} color="var(--accent-green)" />
            </div>
            <div>
              <h3 id="doc-config-title" style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-main)' }}>
                {attachment.name}
              </h3>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                {isPdf ? `${totalPages} pagine • ` : ''}~{tokenCount.toLocaleString()} token stimati
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0.4rem', borderRadius: '6px' }}
            aria-label="Chiudi finestra"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content Body */}
        <div style={{ padding: '1.5rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          
          {/* 1. Page Selection (for PDF) */}
          {isPdf && totalPages > 1 && (
            <div>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.65rem' }}>
                <BookOpen size={16} color="var(--accent-cyan)" /> Selezione Pagine
              </label>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.6rem' }}>
                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  padding: '0.65rem 0.85rem',
                  borderRadius: '8px',
                  background: pageSelection === 'all' ? 'rgba(139, 92, 246, 0.15)' : 'rgba(0,0,0,0.2)',
                  border: `1px solid ${pageSelection === 'all' ? 'var(--accent-purple)' : 'var(--border-color)'}`,
                  cursor: 'pointer',
                  fontSize: '0.82rem'
                }}>
                  <input
                    type="radio"
                    name="pageSelection"
                    checked={pageSelection === 'all'}
                    onChange={() => setPageSelection('all')}
                  />
                  <span>Tutte le pagine (1 - {totalPages})</span>
                </label>

                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  padding: '0.65rem 0.85rem',
                  borderRadius: '8px',
                  background: pageSelection === 'first3' ? 'rgba(139, 92, 246, 0.15)' : 'rgba(0,0,0,0.2)',
                  border: `1px solid ${pageSelection === 'first3' ? 'var(--accent-purple)' : 'var(--border-color)'}`,
                  cursor: 'pointer',
                  fontSize: '0.82rem'
                }}>
                  <input
                    type="radio"
                    name="pageSelection"
                    checked={pageSelection === 'first3'}
                    onChange={() => setPageSelection('first3')}
                  />
                  <span>Prime 3 pagine (1 - {Math.min(3, totalPages)})</span>
                </label>

                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  padding: '0.65rem 0.85rem',
                  borderRadius: '8px',
                  background: pageSelection === 'custom' ? 'rgba(139, 92, 246, 0.15)' : 'rgba(0,0,0,0.2)',
                  border: `1px solid ${pageSelection === 'custom' ? 'var(--accent-purple)' : 'var(--border-color)'}`,
                  cursor: 'pointer',
                  fontSize: '0.82rem'
                }}>
                  <input
                    type="radio"
                    name="pageSelection"
                    checked={pageSelection === 'custom'}
                    onChange={() => setPageSelection('custom')}
                  />
                  <span>Pagine personalizzate</span>
                </label>
              </div>

              {pageSelection === 'custom' && (
                <div style={{ marginTop: '0.65rem' }}>
                  <input
                    type="text"
                    value={customRange}
                    onChange={e => setCustomRange(e.target.value)}
                    placeholder="Es. 1-2, 4, 7"
                    className="input-field"
                    style={{
                      width: '100%',
                      padding: '0.55rem 0.85rem',
                      borderRadius: '8px',
                      background: 'rgba(0,0,0,0.3)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-main)',
                      fontSize: '0.85rem'
                    }}
                  />
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                    Pagine incluse: {includedPages.length > 0 ? includedPages.join(', ') : 'nessuna (intervallo non valido)'}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 2. Smart Cleaning Toggle */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-color)',
            borderRadius: '10px',
            padding: '0.85rem 1rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.85rem'
          }}>
            <input
              type="checkbox"
              id="toggle-smart-clean"
              checked={cleanText}
              onChange={e => setCleanText(e.target.checked)}
              style={{ marginTop: '0.2rem', cursor: 'pointer', width: '16px', height: '16px' }}
            />
            <label htmlFor="toggle-smart-clean" style={{ cursor: 'pointer', flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-main)' }}>
                <Sparkles size={15} color="#eab308" /> Pulizia intelligente del testo
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem', lineHeight: '1.4' }}>
                Rimuove automaticamente disclaimer legali ripetitivi, numeri di pagina e header ridondanti presenti su più pagine. Riduce il tempo di prefill del 50-70% mantenendo tutti i dettagli informativi.
              </div>
            </label>
          </div>

          {/* 3. Live Preview & Token Stats */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                Anteprima testo estratto ({previewText.length.toLocaleString()} caratteri • ~{tokenCount.toLocaleString()} token)
              </span>
              {tokenCount > 2500 && (
                <span style={{ fontSize: '0.74rem', color: '#f59e0b', fontWeight: 500 }}>
                  ⚠️ Documento lungo (richiederà qualche minuto di prefill)
                </span>
              )}
            </div>
            
            <div style={{
              background: 'rgba(0, 0, 0, 0.35)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              padding: '0.75rem',
              maxHeight: '180px',
              overflowY: 'auto',
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: '1.45'
            }}>
              {previewText || '[Nessun testo estratto per la selezione corrente]'}
            </div>
          </div>
        </div>

        {/* Footer Buttons */}
        <div style={{
          padding: '1rem 1.5rem',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '0.75rem',
          background: 'rgba(255, 255, 255, 0.02)'
        }}>
          <button
            onClick={onClose}
            className="btn-secondary"
            style={{ padding: '0.55rem 1rem', fontSize: '0.85rem' }}
          >
            Annulla
          </button>
          <button
            onClick={handleApply}
            className="btn-primary"
            style={{ padding: '0.55rem 1.25rem', fontSize: '0.85rem' }}
          >
            <Check size={16} /> Applica Modifiche
          </button>
        </div>
      </div>
    </div>
  );
}
