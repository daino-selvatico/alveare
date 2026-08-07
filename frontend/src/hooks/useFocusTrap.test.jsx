import React, { useRef } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { useFocusTrap } from './useFocusTrap';

function DummyModal({ isOpen, onClose }) {
  const containerRef = useRef(null);
  useFocusTrap({ isOpen, onClose, containerRef });

  if (!isOpen) return null;

  return (
    <div role="dialog" ref={containerRef}>
      <button id="btn1">First Button</button>
      <button id="btn2">Second Button</button>
    </div>
  );
}

describe('useFocusTrap', () => {
  it('traps focus between first and last element when tab is pressed', () => {
    const handleClose = vi.fn();
    render(<DummyModal isOpen={true} onClose={handleClose} />);

    const btn1 = screen.getByText('First Button');
    const btn2 = screen.getByText('Second Button');

    // Initial focus should be on the first focusable element
    expect(document.activeElement).toBe(btn1);

    // Tab from last element loops to first element
    btn2.focus();
    expect(document.activeElement).toBe(btn2);
    fireEvent.keyDown(window, { key: 'Tab' });
    expect(document.activeElement).toBe(btn1);

    // Shift+Tab from first element loops to last element
    fireEvent.keyDown(window, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(btn2);
  });

  it('calls onClose when Escape key is pressed', () => {
    const handleClose = vi.fn();
    render(<DummyModal isOpen={true} onClose={handleClose} />);

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(handleClose).toHaveBeenCalledTimes(1);
  });
});
