import { describe, it, expect, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { fireEvent } from '@testing-library/react';
import useKeyboardShortcuts from './useKeyboardShortcuts';

describe('useKeyboardShortcuts', () => {
  it('triggers onNewConversation on Ctrl+K', () => {
    const onNewConversation = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onNewConversation }));

    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });
    expect(onNewConversation).toHaveBeenCalledTimes(1);
  });

  it('triggers onToggleShortcutsHelp on Ctrl+/', () => {
    const onToggleShortcutsHelp = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onToggleShortcutsHelp }));

    fireEvent.keyDown(window, { key: '/', ctrlKey: true });
    expect(onToggleShortcutsHelp).toHaveBeenCalledTimes(1);
  });

  it('triggers onToggleShortcutsHelp on ? when not typing in input', () => {
    const onToggleShortcutsHelp = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onToggleShortcutsHelp }));

    fireEvent.keyDown(document.body, { key: '?' });
    expect(onToggleShortcutsHelp).toHaveBeenCalledTimes(1);
  });

  it('does not trigger onToggleShortcutsHelp on ? when typing inside an input', () => {
    const onToggleShortcutsHelp = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onToggleShortcutsHelp }));

    const input = document.createElement('input');
    document.body.appendChild(input);

    fireEvent.keyDown(input, { key: '?' });
    expect(onToggleShortcutsHelp).not.toHaveBeenCalled();

    document.body.removeChild(input);
  });

  it('triggers onEscape on Escape key', () => {
    const onEscape = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onEscape }));

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onEscape).toHaveBeenCalledTimes(1);
  });
});
