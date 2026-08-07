/**
 * LocalStorage utilities for Alveare conversation history management.
 */

const STORAGE_KEY = 'alveare_chat_conversations_v1';
const ACTIVE_CONV_KEY = 'alveare_active_conv_id';

/**
 * @typedef {Object} Message
 * @property {'user' | 'assistant' | 'system'} role
 * @property {string} content
 * @property {number} [timestamp]
 */

/**
 * @typedef {Object} Conversation
 * @property {string} id
 * @property {string} title
 * @property {Message[]} messages
 * @property {string} [systemPrompt]
 * @property {number} createdAt
 * @property {number} updatedAt
 */

export function generateTitleFromMessage(messageContent) {
  if (!messageContent || !messageContent.trim()) {
    return 'Nuova conversazione';
  }
  const cleanText = messageContent.trim().replace(/[\r\n]+/g, ' ');
  if (cleanText.length <= 35) {
    return cleanText;
  }
  return cleanText.slice(0, 35) + '...';
}

export function getConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Sort by updatedAt descending
    return parsed.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  } catch (e) {
    console.error('Failed to load conversations from localStorage:', e);
    return [];
  }
}

export function saveConversations(conversations) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch (e) {
    console.error('Failed to save conversations to localStorage:', e);
  }
}

export function getConversation(id) {
  if (!id) return null;
  const conversations = getConversations();
  return conversations.find(c => c.id === id) || null;
}

export function createConversation(title = 'Nuova conversazione', messages = [], systemPrompt = '') {
  const conversations = getConversations();
  const newConv = {
    id: `conv_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
    title,
    messages,
    systemPrompt,
    createdAt: Date.now(),
    updatedAt: Date.now()
  };

  const updated = [newConv, ...conversations];
  saveConversations(updated);
  setActiveConversationId(newConv.id);
  return newConv;
}

export function saveConversation(conversation) {
  if (!conversation || !conversation.id) return;
  const conversations = getConversations();
  const index = conversations.findIndex(c => c.id === conversation.id);

  const updatedConv = {
    ...conversation,
    updatedAt: Date.now()
  };

  if (index >= 0) {
    conversations[index] = updatedConv;
  } else {
    conversations.unshift(updatedConv);
  }

  saveConversations(conversations);
  return updatedConv;
}

export function deleteConversation(id) {
  if (!id) return;
  const conversations = getConversations();
  const updated = conversations.filter(c => c.id !== id);
  saveConversations(updated);

  const activeId = getActiveConversationId();
  if (activeId === id) {
    const nextActive = updated.length > 0 ? updated[0].id : null;
    setActiveConversationId(nextActive);
  }
}

export function renameConversation(id, newTitle) {
  if (!id || !newTitle || !newTitle.trim()) return;
  const conv = getConversation(id);
  if (conv) {
    conv.title = newTitle.trim();
    saveConversation(conv);
  }
}

export function clearAllConversations() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(ACTIVE_CONV_KEY);
  } catch (e) {
    console.error('Failed to clear conversations:', e);
  }
}

export function getActiveConversationId() {
  try {
    return localStorage.getItem(ACTIVE_CONV_KEY) || null;
  } catch {
    return null;
  }
}

export function setActiveConversationId(id) {
  try {
    if (id) {
      localStorage.setItem(ACTIVE_CONV_KEY, id);
    } else {
      localStorage.removeItem(ACTIVE_CONV_KEY);
    }
  } catch (e) {
    console.error('Failed to set active conversation ID:', e);
  }
}
