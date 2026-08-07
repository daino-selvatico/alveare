import React, { useState, useRef } from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  Edit3,
  Check,
  X,
  Search,
  PanelLeftClose,
  PanelLeft,
  Clock,
  Trash,
  Download,
  Upload
} from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';
import {
  exportConversationsToFile,
  validateConversationsJson,
  importAndMergeConversations
} from '../utils/chatStorage';

function formatDateLabel(timestamp, t) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  const now = new Date();
  
  const isToday = date.toDateString() === now.toDateString();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  const isYesterday = date.toDateString() === yesterday.toDateString();

  if (isToday) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  if (isYesterday) {
    return t('sidebar.yesterday');
  }
  return date.toLocaleDateString([], { day: '2-digit', month: 'short' });
}

export default function SidebarHistory({
  conversations = [],
  activeId = null,
  onSelectConversation,
  onNewConversation,
  onRenameConversation,
  onDeleteConversation,
  onClearAllConversations,
  onImportConversations,
  isCollapsed = false,
  onToggleCollapse
}) {
  const { t } = useTranslation();
  const [editingId, setEditingId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [deletingId, setDeletingId] = useState(null);
  const [importError, setImportError] = useState(null);
  const [importSuccess, setImportSuccess] = useState(null);
  const fileInputRef = useRef(null);

  const startRename = (conv, e) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditingTitle(conv.title);
  };

  const saveRename = (id, e) => {
    if (e) e.stopPropagation();
    if (editingTitle.trim()) {
      onRenameConversation(id, editingTitle.trim());
    }
    setEditingId(null);
    setEditingTitle('');
  };

  const cancelRename = (e) => {
    if (e) e.stopPropagation();
    setEditingId(null);
    setEditingTitle('');
  };

  const confirmDelete = (id, e) => {
    e.stopPropagation();
    setDeletingId(id);
  };

  const executeDelete = (id, e) => {
    e.stopPropagation();
    onDeleteConversation(id);
    setDeletingId(null);
  };

  const handleExportAll = () => {
    exportConversationsToFile(conversations);
  };

  const handleExportSingle = (conv, e) => {
    e.stopPropagation();
    exportConversationsToFile(conv);
  };

  const handleTriggerImport = () => {
    setImportError(null);
    setImportSuccess(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImportError(null);
    setImportSuccess(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const parsed = JSON.parse(text);
        const validation = validateConversationsJson(parsed);

        if (!validation.valid) {
          const key = validation.error || 'importErrorInvalid';
          setImportError(t(`sidebar.${key}`));
          return;
        }

        const merged = importAndMergeConversations(validation.conversations);
        if (onImportConversations) {
          onImportConversations(merged);
        }

        const count = validation.conversations.length;
        setImportSuccess(t('sidebar.importSuccess', { count }));
      } catch {
        setImportError(t('sidebar.importErrorSyntax'));
      } finally {
        if (e.target) {
          e.target.value = '';
        }
      }
    };

    reader.onerror = () => {
      setImportError(t('sidebar.importErrorSyntax'));
      if (e.target) e.target.value = '';
    };

    reader.readAsText(file);
  };

  const filteredConversations = conversations.filter(conv => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return conv.title.toLowerCase().includes(q) ||
      conv.messages.some(m => m.content && m.content.toLowerCase().includes(q));
  });

  if (isCollapsed) {
    return (
      <div style={{
        width: '60px',
        borderRight: '1px solid var(--border-color)',
        background: 'rgba(13, 19, 33, 0.95)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '1rem 0',
        gap: '1rem',
        height: '100%',
        zIndex: 10
      }}>
        <button
          onClick={onToggleCollapse}
          className="btn-secondary"
          style={{ padding: '0.6rem', borderRadius: '10px' }}
          title={t('sidebar.expandSidebar')}
          aria-label={t('sidebar.expandSidebar')}
        >
          <PanelLeft size={18} aria-hidden="true" />
        </button>

        <button
          onClick={onNewConversation}
          className="btn-primary"
          style={{ padding: '0.6rem', borderRadius: '10px' }}
          title={t('sidebar.newChat')}
          aria-label={t('sidebar.newChat')}
        >
          <Plus size={18} aria-hidden="true" />
        </button>

        <div style={{
          width: '80%',
          height: '1px',
          background: 'var(--border-color)',
          margin: '0.2rem 0'
        }} />

        <div style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '0.5rem',
          width: '100%'
        }}>
          {filteredConversations.map(conv => {
            const isActive = conv.id === activeId;
            return (
              <button
                key={conv.id}
                onClick={() => onSelectConversation(conv.id)}
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '10px',
                  border: isActive ? '1px solid var(--accent-purple)' : '1px solid var(--border-color)',
                  background: isActive ? 'rgba(139, 92, 246, 0.25)' : 'rgba(255, 255, 255, 0.03)',
                  color: isActive ? 'white' : 'var(--text-muted)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.2s ease',
                  position: 'relative'
                }}
                title={`${conv.title} (${formatDateLabel(conv.updatedAt, t)})`}
                aria-label={`${conv.title} (${formatDateLabel(conv.updatedAt, t)})`}
                aria-selected={isActive}
              >
                <MessageSquare size={18} color={isActive ? '#c084fc' : 'currentColor'} aria-hidden="true" />
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <aside
      aria-label={t('sidebar.chatHistory')}
      style={{
        width: '280px',
        minWidth: '280px',
        borderRight: '1px solid var(--border-color)',
        background: 'rgba(13, 19, 33, 0.95)',
        backdropFilter: 'blur(12px)',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        zIndex: 10,
        userSelect: 'none'
      }}
    >
      {/* Header */}
      <div style={{
        padding: '1rem',
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.85rem'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.92rem', color: 'var(--text-main)' }}>
            <Clock size={16} color="var(--accent-purple)" aria-hidden="true" />
            <span>{t('sidebar.chatHistory')}</span>
            <span style={{
              fontSize: '0.72rem',
              padding: '0.1rem 0.45rem',
              borderRadius: '999px',
              background: 'rgba(139, 92, 246, 0.15)',
              color: '#c084fc',
              fontWeight: 600
            }}>
              {conversations.length}
            </span>
          </div>

          <button
            onClick={onToggleCollapse}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '0.3rem',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center'
            }}
            title={t('sidebar.collapseSidebar')}
            aria-label={t('sidebar.collapseSidebar')}
          >
            <PanelLeftClose size={18} aria-hidden="true" />
          </button>
        </div>

        {/* New Chat Button */}
        <button
          onClick={onNewConversation}
          className="btn-primary"
          style={{
            width: '100%',
            justifyContent: 'center',
            padding: '0.65rem',
            fontSize: '0.88rem',
            borderRadius: '9px'
          }}
          aria-label={t('sidebar.newChat')}
        >
          <Plus size={18} aria-hidden="true" /> {t('sidebar.newChat')}
        </button>

        {/* Search Input */}
        {conversations.length > 3 && (
          <div style={{ position: 'relative' }}>
            <Search size={14} style={{ position: 'absolute', left: '0.65rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} aria-hidden="true" />
            <input
              type="text"
              placeholder={t('sidebar.searchPlaceholder')}
              aria-label={t('sidebar.searchPlaceholder')}
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                padding: '0.4rem 0.6rem 0.4rem 2rem',
                fontSize: '0.8rem',
                borderRadius: '7px',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-color)',
                color: 'white',
                outline: 'none'
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute',
                  right: '0.5rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer'
                }}
                aria-label="Cancella ricerca"
              >
                <X size={12} aria-hidden="true" />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Conversations List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0.65rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.35rem'
      }}>
        {filteredConversations.length === 0 ? (
          <div style={{
            padding: '2rem 1rem',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: '0.83rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <MessageSquare size={24} style={{ opacity: 0.4 }} aria-hidden="true" />
            {searchQuery ? t('sidebar.noConversationsFound') : t('sidebar.noConversationsSaved')}
          </div>
        ) : (
          filteredConversations.map(conv => {
            const isActive = conv.id === activeId;
            const isEditing = editingId === conv.id;
            const isDeleting = deletingId === conv.id;
            const count = conv.messages ? conv.messages.length : 0;
            const countStr = count === 1 ? t('sidebar.messagesCount_one', { count: 1 }) : t('sidebar.messagesCount_other', { count });

            return (
              <div
                key={conv.id}
                role="button"
                tabIndex={0}
                aria-label={`${conv.title || t('sidebar.defaultTitle')} (${formatDateLabel(conv.updatedAt, t)})`}
                aria-selected={isActive}
                onClick={() => !isEditing && onSelectConversation(conv.id)}
                onKeyDown={e => {
                  if ((e.key === 'Enter' || e.key === ' ') && !isEditing) {
                    e.preventDefault();
                    onSelectConversation(conv.id);
                  }
                }}
                style={{
                  padding: '0.65rem 0.75rem',
                  borderRadius: '9px',
                  background: isActive
                    ? 'linear-gradient(90deg, rgba(139, 92, 246, 0.22) 0%, rgba(6, 182, 212, 0.12) 100%)'
                    : 'transparent',
                  border: isActive
                    ? '1px solid rgba(139, 92, 246, 0.4)'
                    : '1px solid transparent',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '0.5rem',
                  transition: 'all 0.15rem ease',
                  position: 'relative'
                }}
                className="sidebar-conv-item"
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flex: 1, minWidth: 0 }}>
                  <MessageSquare size={16} color={isActive ? '#c084fc' : 'var(--text-muted)'} style={{ flexShrink: 0 }} aria-hidden="true" />

                  {isEditing ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', width: '100%' }} onClick={e => e.stopPropagation()}>
                      <input
                        type="text"
                        value={editingTitle}
                        onChange={e => setEditingTitle(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') saveRename(conv.id, e);
                          if (e.key === 'Escape') cancelRename(e);
                        }}
                        autoFocus
                        aria-label="Nuovo titolo conversazione"
                        style={{
                          width: '100%',
                          padding: '0.2rem 0.4rem',
                          fontSize: '0.82rem',
                          borderRadius: '4px',
                          background: 'rgba(0,0,0,0.5)',
                          border: '1px solid var(--accent-purple)',
                          color: 'white',
                          outline: 'none'
                        }}
                      />
                      <button
                        onClick={e => saveRename(conv.id, e)}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-green)', cursor: 'pointer', padding: '0.2rem' }}
                        title={t('sidebar.save')}
                        aria-label={t('sidebar.save')}
                      >
                        <Check size={14} aria-hidden="true" />
                      </button>
                      <button
                        onClick={cancelRename}
                        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0.2rem' }}
                        title={t('sidebar.cancel')}
                        aria-label={t('sidebar.cancel')}
                      >
                        <X size={14} aria-hidden="true" />
                      </button>
                    </div>
                  ) : (
                    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                      <span style={{
                        fontSize: '0.84rem',
                        fontWeight: isActive ? 600 : 400,
                        color: isActive ? 'white' : 'var(--text-main)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {conv.title || t('sidebar.defaultTitle')}
                      </span>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        {formatDateLabel(conv.updatedAt, t)} • {countStr}
                      </span>
                    </div>
                  )}
                </div>

                {/* Actions (Rename & Delete) */}
                {!isEditing && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }} className="conv-actions">
                    {isDeleting ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }} onClick={e => e.stopPropagation()}>
                        <button
                          onClick={e => executeDelete(conv.id, e)}
                          style={{ background: 'var(--accent-red)', border: 'none', borderRadius: '4px', color: 'white', cursor: 'pointer', padding: '0.15rem 0.4rem', fontSize: '0.7rem', fontWeight: 600 }}
                          title={t('sidebar.confirmDelete')}
                          aria-label={t('sidebar.confirmDelete')}
                        >
                          {t('sidebar.confirmDelete')}
                        </button>
                        <button
                          onClick={e => { e.stopPropagation(); setDeletingId(null); }}
                          style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0.15rem' }}
                          title={t('sidebar.cancel')}
                          aria-label={t('sidebar.cancel')}
                        >
                          <X size={13} aria-hidden="true" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <button
                          onClick={e => handleExportSingle(conv, e)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            padding: '0.25rem',
                            borderRadius: '4px',
                            transition: 'color 0.2s ease'
                          }}
                          title={t('sidebar.exportSelected')}
                          aria-label={t('sidebar.exportSelected')}
                        >
                          <Download size={13} aria-hidden="true" />
                        </button>

                        <button
                          onClick={e => startRename(conv, e)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            padding: '0.25rem',
                            borderRadius: '4px',
                            transition: 'color 0.2s ease'
                          }}
                          title={t('sidebar.rename')}
                          aria-label={t('sidebar.rename')}
                        >
                          <Edit3 size={13} aria-hidden="true" />
                        </button>

                        <button
                          onClick={e => confirmDelete(conv.id, e)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            padding: '0.25rem',
                            borderRadius: '4px',
                            transition: 'color 0.2s ease'
                          }}
                          title={t('sidebar.delete')}
                          aria-label={t('sidebar.delete')}
                        >
                          <Trash2 size={13} aria-hidden="true" />
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer / Import & Export & Clear All options */}
      <div style={{
        padding: '0.75rem 1rem',
        borderTop: '1px solid var(--border-color)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".json,application/json"
          style={{ display: 'none' }}
          aria-hidden="true"
        />

        <div style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
          <button
            onClick={handleExportAll}
            disabled={conversations.length === 0}
            className="btn-secondary"
            style={{
              flex: 1,
              padding: '0.4rem 0.5rem',
              fontSize: '0.78rem',
              justifyContent: 'center',
              gap: '0.35rem',
              opacity: conversations.length === 0 ? 0.5 : 1,
              cursor: conversations.length === 0 ? 'not-allowed' : 'pointer'
            }}
            title={t('sidebar.exportTitle')}
            aria-label={t('sidebar.exportTitle')}
          >
            <Download size={14} aria-hidden="true" /> {t('sidebar.export')}
          </button>

          <button
            onClick={handleTriggerImport}
            className="btn-secondary"
            style={{
              flex: 1,
              padding: '0.4rem 0.5rem',
              fontSize: '0.78rem',
              justifyContent: 'center',
              gap: '0.35rem'
            }}
            title={t('sidebar.importTitle')}
            aria-label={t('sidebar.importTitle')}
          >
            <Upload size={14} aria-hidden="true" /> {t('sidebar.import')}
          </button>
        </div>

        {importError && (
          <div
            role="alert"
            aria-live="polite"
            style={{
              fontSize: '0.75rem',
              color: '#f87171',
              background: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '6px',
              padding: '0.4rem 0.6rem',
              display: 'flex',
              alignItems: 'center',
              justify: 'space-between',
              gap: '0.3rem'
            }}
          >
            <span>{importError}</span>
            <button
              onClick={() => setImportError(null)}
              style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer', padding: 0 }}
              aria-label={t('sidebar.cancel')}
            >
              <X size={12} aria-hidden="true" />
            </button>
          </div>
        )}

        {importSuccess && (
          <div
            role="status"
            aria-live="polite"
            style={{
              fontSize: '0.75rem',
              color: '#4ade80',
              background: 'rgba(74, 222, 128, 0.15)',
              border: '1px solid rgba(74, 222, 128, 0.3)',
              borderRadius: '6px',
              padding: '0.4rem 0.6rem',
              display: 'flex',
              alignItems: 'center',
              justify: 'space-between',
              gap: '0.3rem'
            }}
          >
            <span>{importSuccess}</span>
            <button
              onClick={() => setImportSuccess(null)}
              style={{ background: 'none', border: 'none', color: '#4ade80', cursor: 'pointer', padding: 0 }}
              aria-label={t('sidebar.cancel')}
            >
              <X size={12} aria-hidden="true" />
            </button>
          </div>
        )}

        {conversations.length > 0 && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '0.2rem' }}>
            <button
              onClick={() => {
                if (window.confirm(t('sidebar.deleteConfirm'))) {
                  onClearAllConversations();
                }
              }}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: '0.76rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                transition: 'color 0.2s ease'
              }}
              title={t('sidebar.clearHistoryTitle')}
              aria-label={t('sidebar.clearHistoryTitle')}
            >
              <Trash size={13} aria-hidden="true" /> {t('sidebar.clearHistory')}
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}

