import React, { useState } from 'react';
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
  Trash
} from 'lucide-react';
import { useTranslation } from '../i18n/I18nContext';

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
  isCollapsed = false,
  onToggleCollapse
}) {
  const { t } = useTranslation();
  const [editingId, setEditingId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [deletingId, setDeletingId] = useState(null);

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
        >
          <PanelLeft size={18} />
        </button>

        <button
          onClick={onNewConversation}
          className="btn-primary"
          style={{ padding: '0.6rem', borderRadius: '10px' }}
          title={t('sidebar.newChat')}
        >
          <Plus size={18} />
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
              >
                <MessageSquare size={18} color={isActive ? '#c084fc' : 'currentColor'} />
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <aside style={{
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
    }}>
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
            <Clock size={16} color="var(--accent-purple)" />
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
          >
            <PanelLeftClose size={18} />
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
        >
          <Plus size={18} /> {t('sidebar.newChat')}
        </button>

        {/* Search Input */}
        {conversations.length > 3 && (
          <div style={{ position: 'relative' }}>
            <Search size={14} style={{ position: 'absolute', left: '0.65rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder={t('sidebar.searchPlaceholder')}
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
              >
                <X size={12} />
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
            <MessageSquare size={24} style={{ opacity: 0.4 }} />
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
                onClick={() => !isEditing && onSelectConversation(conv.id)}
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
                  <MessageSquare size={16} color={isActive ? '#c084fc' : 'var(--text-muted)'} style={{ flexShrink: 0 }} />

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
                      >
                        <Check size={14} />
                      </button>
                      <button
                        onClick={cancelRename}
                        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0.2rem' }}
                        title={t('sidebar.cancel')}
                      >
                        <X size={14} />
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
                        >
                          {t('sidebar.confirmDelete')}
                        </button>
                        <button
                          onClick={e => { e.stopPropagation(); setDeletingId(null); }}
                          style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0.15rem' }}
                          title={t('sidebar.cancel')}
                        >
                          <X size={13} />
                        </button>
                      </div>
                    ) : (
                      <>
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
                        >
                          <Edit3 size={13} />
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
                        >
                          <Trash2 size={13} />
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

      {/* Footer / Clear All option */}
      {conversations.length > 0 && (
        <div style={{
          padding: '0.75rem 1rem',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center'
        }}>
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
          >
            <Trash size={13} /> {t('sidebar.clearHistory')}
          </button>
        </div>
      )}
    </aside>
  );
}

