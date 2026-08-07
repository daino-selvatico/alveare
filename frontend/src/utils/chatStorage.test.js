import { describe, it, expect, beforeEach } from 'vitest';
import {
  DEFAULT_SETTINGS,
  getGlobalSettings,
  saveGlobalSettings,
  generateTitleFromMessage,
  getConversations,
  saveConversations,
  getConversation,
  createConversation,
  saveConversation,
  deleteConversation,
  renameConversation,
  clearAllConversations,
  getActiveConversationId,
  setActiveConversationId,
} from './chatStorage';

describe('chatStorage utility', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe('generateTitleFromMessage', () => {
    it('returns default title for empty or whitespace string', () => {
      expect(generateTitleFromMessage('')).toBe('Nuova conversazione');
      expect(generateTitleFromMessage('   ')).toBe('Nuova conversazione');
      expect(generateTitleFromMessage(null)).toBe('Nuova conversazione');
      expect(generateTitleFromMessage(undefined)).toBe('Nuova conversazione');
    });

    it('returns original message when length is 35 characters or fewer', () => {
      const shortMsg = 'Ciao, come stai?';
      expect(generateTitleFromMessage(shortMsg)).toBe(shortMsg);

      const exact35 = '12345678901234567890123456789012345';
      expect(generateTitleFromMessage(exact35)).toBe(exact35);
    });

    it('truncates message and appends ellipsis when longer than 35 characters', () => {
      const longMsg = 'Questo è un messaggio molto lungo che supera il limite dei trentacinque caratteri.';
      const expected = longMsg.slice(0, 35) + '...';
      expect(generateTitleFromMessage(longMsg)).toBe(expected);
    });

    it('replaces newlines with spaces', () => {
      const multiline = 'Prima riga\nSeconda riga\r\nTerza riga';
      expect(generateTitleFromMessage(multiline)).toBe('Prima riga Seconda riga Terza riga');
    });
  });

  describe('Global Settings', () => {
    it('returns default settings when none stored', () => {
      const settings = getGlobalSettings();
      expect(settings).toEqual(DEFAULT_SETTINGS);
    });

    it('saves and merges global settings', () => {
      saveGlobalSettings({ temperature: 0.2, enableThinking: false });
      const settings = getGlobalSettings();
      expect(settings.temperature).toBe(0.2);
      expect(settings.enableThinking).toBe(false);
      expect(settings.systemPrompt).toBe(DEFAULT_SETTINGS.systemPrompt);
    });
  });

  describe('Active Conversation Persistence', () => {
    it('returns null when no active conversation set', () => {
      expect(getActiveConversationId()).toBeNull();
    });

    it('sets and retrieves active conversation ID', () => {
      setActiveConversationId('conv_123');
      expect(getActiveConversationId()).toBe('conv_123');
    });

    it('removes active conversation ID when set to null or empty', () => {
      setActiveConversationId('conv_123');
      setActiveConversationId(null);
      expect(getActiveConversationId()).toBeNull();
    });
  });

  describe('Conversations CRUD Operations', () => {
    it('returns empty array when no conversations stored', () => {
      expect(getConversations()).toEqual([]);
    });

    it('creates a new conversation and sets it as active', () => {
      const conv = createConversation('Test Conv', [{ role: 'user', content: 'Hello' }]);

      expect(conv.id).toBeDefined();
      expect(conv.title).toBe('Test Conv');
      expect(conv.messages).toHaveLength(1);
      expect(conv.messages[0].content).toBe('Hello');

      const all = getConversations();
      expect(all).toHaveLength(1);
      expect(all[0].id).toBe(conv.id);
      expect(getActiveConversationId()).toBe(conv.id);
    });

    it('saves conversations directly with saveConversations', () => {
      const convList = [{ id: 'manual_1', title: 'Manual', messages: [], createdAt: 100, updatedAt: 200 }];
      saveConversations(convList);
      expect(getConversations()).toHaveLength(1);
      expect(getConversation('manual_1')?.title).toBe('Manual');
    });

    it('retrieves a conversation by ID', () => {
      const conv = createConversation('Find Me');
      const retrieved = getConversation(conv.id);
      expect(retrieved).not.toBeNull();
      expect(retrieved?.title).toBe('Find Me');

      expect(getConversation('non_existent_id')).toBeNull();
    });

    it('renames a conversation', () => {
      const conv = createConversation('Old Title');
      renameConversation(conv.id, 'New Title');

      const updated = getConversation(conv.id);
      expect(updated?.title).toBe('New Title');
    });

    it('ignores invalid rename parameters', () => {
      const conv = createConversation('Original Title');
      renameConversation(conv.id, '   ');
      expect(getConversation(conv.id)?.title).toBe('Original Title');

      renameConversation('invalid_id', 'New Title');
    });

    it('updates existing conversation with saveConversation', () => {
      const conv = createConversation('Initial');
      conv.messages.push({ role: 'assistant', content: 'Reply' });

      saveConversation(conv);

      const saved = getConversation(conv.id);
      expect(saved?.messages).toHaveLength(1);
      expect(saved?.messages[0].content).toBe('Reply');
    });

    it('deletes a conversation and updates active ID if deleted was active', () => {
      const conv1 = createConversation('Conv 1');
      const conv2 = createConversation('Conv 2');

      // conv2 was created last, so active ID should be conv2.id
      expect(getActiveConversationId()).toBe(conv2.id);

      deleteConversation(conv2.id);

      const remaining = getConversations();
      expect(remaining).toHaveLength(1);
      expect(remaining[0].id).toBe(conv1.id);
      // Active ID should shift to remaining conversation
      expect(getActiveConversationId()).toBe(conv1.id);
    });

    it('clears all conversations and active conversation ID', () => {
      createConversation('Conv 1');
      createConversation('Conv 2');

      clearAllConversations();

      expect(getConversations()).toEqual([]);
      expect(getActiveConversationId()).toBeNull();
    });
  });
});
