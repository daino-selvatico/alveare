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
  PanelLeftOpen, 
  Clock, 
  AlertTriangle 
} from 'lucide-react';

export default function SidebarHistory({
  conversations,
  activeId,
  onSelectConversation,
  onNewConversation,
  onRenameConversation,
  onDeleteConversation,
  onClearAll,
  isOpen,
  onToggleOpen
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const filteredConversations = conversations
    .filter(conv => {
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      const titleMatch = conv.title.toLowerCase().includes(q);
      const msgMatch = conv.messages?.some(m => m.content?.toLowerCase().includes(q));
      return titleMatch || msgMatch;
    })
    .sort((a, b) => b.updatedAt - a.updatedAt);

  const handleStartRename = (e, conv) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditingTitle(conv.title);
  };

  const handleSaveRename = (e, id) => {
    e.stopPropagation();
    if (editingTitle.trim()) {
      onRenameConversation(id, editingTitle.trim());
    }
    setEditingId(null);
  };

  const handleCancelRename = (e) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const handleDelete = (e, id) => {
    e.stopPropagation();
    onDeleteConversation(id);
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    
    if (isToday) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  if (!isOpen) {
    return (
      <div style={{
        width: '52px',
        borderRight: '1px solid var(--border-color)',
        background: 'rgba(11, 15, 25, 0.95)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '0.85rem 0',
        gap: '1rem',
        zIndex: 20
      }}>
        <button
          onClick={onToggleOpen}
          className="btn-secondary"
          style={{ padding: '0.5rem', borderRadius: '8px', border: 'none', color: 'var(--text-muted)' }}
          title="Espandi Storico"
        >
          <PanelLeftOpen size={20} />
        </button>

        <button
          onClick={onNewConversation}
          style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'var(--gradient-brand)',
            border: 'none',
            color: 'white',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 12px rgba(139, 92, 246, 0.4)'
          }}
          title="Nuova Conversazione"
        >
          <Plus size={20} />
        </button>
      </div>
    );
  }

  return (
    <aside style={{
      width: '280px',
      borderRight: '1px solid var(--border-color)',
      background: 'rgba(11, 15, 25, 0.95)',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      zIndex: 20,
      userSelect: 'none'
    }}>
      {/* Header */}
      <div style={{
        padding: '0.85rem 1rem',
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Clock size={18} color="var(--accent-purple)" />
          <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'white' }}>
            Storico Chat
          </span>
          <span className="badge" style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#c084fc', padding: '0.1rem 0.45rem', fontSize: '0.7rem' }}>
            {conversations.length}
          </span>
        </div>

        <button
          onClick={onToggleOpen}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            padding: '0.25rem',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center'
          }}
          title="Nascondi Sidebar"
        >
          <PanelLeftClose size={18} />
        </button>
      </div>

      {/* New Chat Button */}
      <div style={{ padding: '0.85rem 1rem 0.5rem 1rem' }}>
        <button
          onClick={onNewConversation}
          className="btn-primary"
          style={{
            width: '100%',
            justifyContent: 'center',
            padding: '0.65rem 1rem',
            fontSize: '0.9rem',
            fontWeight: 600,
            borderRadius: '10px'
          }}
        >
          <Plus size={18} /> Nuova Conversazione
        </button>
      </div>

      {/* Search Input */}
      {conversations.length > 2 && (
        <div style={{ padding: '0.4rem 1rem 0.6rem 1rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '0.35rem 0.65rem',
            gap: '0.4rem'
          }}>
            <Search size={14} color="var(--text-muted)" />
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Cerca conversazione..."
              style={{
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: 'white',
                fontSize: '0.82rem',
                width: '100%'
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={12} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Conversation List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0.5rem 0.75rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.35rem'
      }}>
        {filteredConversations.length === 0 ? (
          <div style={{
            padding: '2rem 1rem',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: '0.83rem'
          }}>
            {searchQuery ? 'Nessuna conversazione trovata.' : 'Nessuna conversazione trovata.'}
          </div>
        ) : (
          filteredConversations.map(conv => {
            const isActive = conv.id === activeId;
            const isEditing = editingId === conv.id;
            const userMsgCount = conv.messages ? conv.messages.filter(m => m.role === 'user').length : 0;

            return (
              <div
                key={conv.id}
                onClick={() => !isEditing && onSelectConversation(conv.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.6rem 0.75rem',
                  borderRadius: '9px',
                  cursor: 'pointer',
                  background: isActive ? 'rgba(139, 92, 246, 0.18)' : 'rgba(255, 255, 255, 0.02)',
                  border: isActive ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid transparent',
                  borderLeft: isActive ? '3px solid var(--accent-purple)' : '1px solid transparent',
                  transition: 'all 0.15rem ease',
                  position: 'relative'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', minWidth: 0, flex: 1 }}>
                  <MessageSquare size={16} color={isActive ? '#c084fc' : 'var(--text-muted)'} style={{ flexShrink: 0 }} />
                  
                  {isEditing ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', width: '100%' }}>
                      <input
                        type="text"
                        value={editingTitle}
                        onChange={e => setEditingTitle(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') handleSaveRename(e, conv.id);
                          if (e.key === 'Escape') handleCancelRename(e);
                        }}
                        autoFocus
                        onClick={e => e.stopPropagation()}
                        style={{
                          background: 'rgba(0,0,0,0.5)',
                          border: '1px solid var(--accent-purple)',
                          borderRadius: '4px',
                          color: 'white',
                          fontSize: '0.83rem',
                          padding: '0.15rem 0.4rem',
                          width: '100%',
                          outline: 'none'
                        }}
                      />
                      <button
                        onClick={e => handleSaveRename(e, conv.id)}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-green)', cursor: 'pointer', padding: '0.1rem' }}
                      >
                        <Check size={14} />
                      </button>
                      <button
                        onClick={handleCancelRename}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-red)', cursor: 'pointer', padding: '0.1rem' }}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0, flex: 1 }}>
                      <span style={{
                        color: isActive ? 'white' : 'var(--text-main)',
                        fontSize: '0.85rem',
                        fontWeight: isActive ? 600 : 400,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {conv.title}
                      </span>
                      
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        <span>{formatDate(conv.updatedAt)}</span>
                        {userMsgCount > 0 && (
                          <span>• {userMsgCount} {userMsgCount === 1 ? 'turno' : 'turni'}</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {!isEditing && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.2rem',
                    opacity: isActive ? 1 : 0.6,
                    transition: 'opacity 0.15s ease'
                  }}>
                    <button
                      onClick={e => handleStartRename(e, conv)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--text-muted)',
                        cursor: 'pointer',
                        padding: '0.25rem',
                        borderRadius: '4px'
                      }}
                      title="Rinomina"
                    >
                      <Edit3 size={13} />
                    </button>
                    <button
                      onClick={e => handleDelete(e, conv.id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--text-muted)',
                        cursor: 'pointer',
                        padding: '0.25rem',
                        borderRadius: '4px'
                      }}
                      title="Elimina conversazione"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer / Clear All */}
      <div style={{
        padding: '0.75rem 1rem',
        borderTop: '1px solid var(--border-color)',
        background: 'rgba(0, 0, 0, 0.2)'
      }}>
        {showClearConfirm ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ fontSize: '0.78rem', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <AlertTriangle size={14} /> Eliminare tutto lo storico?
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn-danger"
                style={{ flex: 1, padding: '0.35rem 0.5rem', fontSize: '0.78rem', justifyContent: 'center' }}
                onClick={() => {
                  onClearAll();
                  setShowClearConfirm(false);
                }}
              >
                Sì, Elimina
              </button>
              <button
                className="btn-secondary"
                style={{ flex: 1, padding: '0.35rem 0.5rem', fontSize: '0.78rem', justifyContent: 'center' }}
                onClick={() => setShowClearConfirm(false)}
              >
                Annulla
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowClearConfirm(true)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              fontSize: '0.8rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              width: '100%',
              padding: '0.3rem 0'
            }}
          >
            <Trash2 size={14} /> Cancella tutto lo storico
          </button>
        )}
      </div>
    </aside>
  );
}
