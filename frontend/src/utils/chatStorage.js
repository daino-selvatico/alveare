// Utility module for persisting conversation history & active conversation state in localStorage

const STORAGE_KEY_CONVERSATIONS = 'alveare_conversations';
const STORAGE_KEY_ACTIVE_ID = 'alveare_active_conv_id';

export const DEFAULT_WELCOME_MSG = {
  role: 'assistant',
  content: "Ciao! Sono il modello LLM in esecuzione sull'NPU AMD Ryzen AI. Come posso aiutarti oggi?"
};

export function createNewConversation(title = 'Nuova conversazione') {
  const timestamp = Date.now();
  return {
    id: `conv_${timestamp}_${Math.random().toString(36).substring(2, 7)}`,
    title,
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [{ ...DEFAULT_WELCOME_MSG }],
    systemPrompt: 'Sei un assistente AI esperto ed utile.'
  };
}

export function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_CONVERSATIONS);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed;
      }
    }
  } catch (e) {
    console.error('Failed to load conversations from localStorage:', e);
  }
  const initialConv = createNewConversation();
  saveConversations([initialConv]);
  saveActiveId(initialConv.id);
  return [initialConv];
}

export function saveConversations(conversations) {
  try {
    localStorage.setItem(STORAGE_KEY_CONVERSATIONS, JSON.stringify(conversations));
  } catch (e) {
    console.error('Failed to save conversations to localStorage:', e);
  }
}

export function loadActiveId() {
  try {
    return localStorage.getItem(STORAGE_KEY_ACTIVE_ID) || null;
  } catch (e) {
    return null;
  }
}

export function saveActiveId(id) {
  try {
    if (id) {
      localStorage.setItem(STORAGE_KEY_ACTIVE_ID, id);
    } else {
      localStorage.removeItem(STORAGE_KEY_ACTIVE_ID);
    }
  } catch (e) {
    console.error('Failed to save active conversation id:', e);
  }
}

export function generateTitleFromMessage(userMessage) {
  if (!userMessage) return 'Nuova conversazione';
  const clean = userMessage.trim().replace(/[\r\n]+/g, ' ');
  if (clean.length <= 32) {
    return clean;
  }
  return clean.substring(0, 32) + '...';
}
