import { useEffect } from 'react';

function isTypingInInput(target) {
  if (!target) return false;
  const tagName = target.tagName;
  return (
    tagName === 'INPUT' ||
    tagName === 'TEXTAREA' ||
    tagName === 'SELECT' ||
    target.isContentEditable
  );
}

export function useKeyboardShortcuts({
  onNewConversation,
  onToggleShortcutsHelp,
  onEscape
}) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      const isMod = e.ctrlKey || e.metaKey;
      const key = e.key;

      // 1. Ctrl/Cmd + K: New Conversation
      if (isMod && (key.toLowerCase() === 'k' || e.code === 'KeyK')) {
        e.preventDefault();
        onNewConversation?.();
        return;
      }

      // 2. Ctrl/Cmd + /: Toggle shortcuts help modal
      if (isMod && (key === '/' || e.code === 'Slash')) {
        e.preventDefault();
        onToggleShortcutsHelp?.();
        return;
      }

      // 3. '?' key when not typing in an input field
      if (key === '?' && !isTypingInInput(e.target)) {
        e.preventDefault();
        onToggleShortcutsHelp?.();
        return;
      }

      // 4. Esc key: Stop generation or close open overlays
      if (key === 'Escape') {
        onEscape?.(e);
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNewConversation, onToggleShortcutsHelp, onEscape]);
}

export default useKeyboardShortcuts;
