import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import KeyboardShortcutsModal from './KeyboardShortcutsModal';
import { I18nProvider } from '../i18n/I18nContext';

describe('KeyboardShortcutsModal', () => {
  it('does not render when isOpen is false', () => {
    render(
      <I18nProvider>
        <KeyboardShortcutsModal isOpen={false} onClose={vi.fn()} />
      </I18nProvider>
    );

    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('renders modal content when isOpen is true', () => {
    render(
      <I18nProvider>
        <KeyboardShortcutsModal isOpen={true} onClose={vi.fn()} />
      </I18nProvider>
    );

    expect(screen.getByRole('dialog')).toBeDefined();
    expect(screen.getAllByText('Enter').length).toBeGreaterThan(0);
    expect(screen.getByText('Esc')).toBeDefined();
  });

  it('calls onClose when close button is clicked', () => {
    const handleClose = vi.fn();
    render(
      <I18nProvider>
        <KeyboardShortcutsModal isOpen={true} onClose={handleClose} />
      </I18nProvider>
    );

    const closeBtn = screen.getByLabelText(/close|chiudi/i);
    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when Escape key is pressed', () => {
    const handleClose = vi.fn();
    render(
      <I18nProvider>
        <KeyboardShortcutsModal isOpen={true} onClose={handleClose} />
      </I18nProvider>
    );

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(handleClose).toHaveBeenCalledTimes(1);
  });
});
