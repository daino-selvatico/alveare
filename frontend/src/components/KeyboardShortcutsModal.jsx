import React, { useRef } from 'react';
import { X, Keyboard, CornerDownLeft, ArrowDown, OctagonAlert, PlusCircle, HelpCircle } from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';
import { useFocusTrap } from '../hooks/useFocusTrap';

export default function KeyboardShortcutsModal({ isOpen, onClose }) {
  const { t } = useTranslation();
  const containerRef = useRef(null);

  useFocusTrap({ isOpen, onClose, containerRef });

  if (!isOpen) return null;

  const isMac = typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
  const modKey = isMac ? '⌘' : 'Ctrl';

  const shortcutList = [
    {
      keys: ['Enter'],
      icon: <CornerDownLeft size={16} color="var(--accent-purple)" aria-hidden="true" />,
      label: t('shortcuts.sendMsg'),
      scope: t('shortcuts.sendMsgScope')
    },
    {
      keys: ['Shift', 'Enter'],
      icon: <ArrowDown size={16} color="var(--accent-cyan)" aria-hidden="true" />,
      label: t('shortcuts.newLine'),
      scope: t('shortcuts.newLineScope')
    },
    {
      keys: ['Esc'],
      icon: <OctagonAlert size={16} color="var(--accent-red)" aria-hidden="true" />,
      label: t('shortcuts.stopGen'),
      scope: t('shortcuts.stopGenScope')
    },
    {
      keys: [modKey, 'K'],
      icon: <PlusCircle size={16} color="var(--accent-green)" aria-hidden="true" />,
      label: t('shortcuts.newChat'),
      scope: t('shortcuts.newChatScope')
    },
    {
      keys: [modKey, '/'],
      badgeAlt: '?',
      icon: <HelpCircle size={16} color="var(--accent-amber)" aria-hidden="true" />,
      label: t('shortcuts.toggleHelp'),
      scope: t('shortcuts.toggleHelpScope')
    }
  ];

  return (
    <div
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="shortcuts-modal-title"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.75)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '1.5rem',
        animation: 'fadeIn 0.2s ease-out'
      }}
    >
      <div
        ref={containerRef}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          maxWidth: '520px',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '16px',
          boxShadow: '0 20px 50px rgba(0, 0, 0, 0.5), var(--shadow-glow)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '1.25rem 1.5rem',
            borderBottom: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'rgba(255, 255, 255, 0.02)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '10px',
                background: 'rgba(139, 92, 246, 0.15)',
                border: '1px solid rgba(139, 92, 246, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Keyboard size={20} color="var(--accent-purple)" aria-hidden="true" />
            </div>
            <div>
              <h3 id="shortcuts-modal-title" style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>
                {t('shortcuts.title')}
              </h3>
            </div>
          </div>

          <button
            onClick={onClose}
            className="btn-secondary"
            style={{
              padding: '0.4rem',
              borderRadius: '8px',
              border: 'none',
              background: 'transparent',
              color: 'var(--text-muted)',
              cursor: 'pointer'
            }}
            aria-label={t('shortcuts.close')}
          >
            <X size={20} aria-hidden="true" />
          </button>
        </div>

        {/* Content list */}
        <div style={{ padding: '1.25rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {shortcutList.map((item, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.75rem 1rem',
                borderRadius: '10px',
                background: 'rgba(0, 0, 0, 0.2)',
                border: '1px solid var(--border-color)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {item.icon}
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-main)' }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-subtle)' }}>
                    {item.scope}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                {item.keys.map((key, kIdx) => (
                  <React.Fragment key={kIdx}>
                    <kbd
                      style={{
                        background: 'rgba(255, 255, 255, 0.08)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '6px',
                        padding: '0.25rem 0.55rem',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.82rem',
                        fontWeight: 600,
                        color: 'var(--text-main)',
                        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
                      }}
                    >
                      {key}
                    </kbd>
                    {kIdx < item.keys.length - 1 && (
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>+</span>
                    )}
                  </React.Fragment>
                ))}
                {item.badgeAlt && (
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-subtle)', marginLeft: '0.25rem' }}>
                    ({t('sidebar.cancel') ? 'o' : 'or'} <kbd style={{
                      background: 'rgba(255, 255, 255, 0.08)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      padding: '0.25rem 0.45rem',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.82rem',
                      fontWeight: 600,
                      color: 'var(--text-main)'
                    }}>{item.badgeAlt}</kbd>)
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '1rem 1.5rem',
            borderTop: '1px solid var(--border-color)',
            display: 'flex',
            justify: 'flex-end',
            background: 'rgba(0, 0, 0, 0.15)'
          }}
        >
          <button
            className="btn-primary"
            onClick={onClose}
            style={{ padding: '0.5rem 1.25rem', borderRadius: '8px', fontSize: '0.88rem' }}
          >
            {t('shortcuts.close')}
          </button>
        </div>
      </div>
    </div>
  );
}
