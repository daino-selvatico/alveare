import { useEffect, useRef } from 'react';

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function useFocusTrap({ isOpen, onClose, containerRef }) {
  const previousFocusRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;

    // Store currently focused element to return focus later
    previousFocusRef.current = document.activeElement;

    const container = containerRef?.current;
    if (container) {
      const focusables = Array.from(container.querySelectorAll(FOCUSABLE_SELECTOR));
      if (focusables.length > 0) {
        // Focus first element or container
        focusables[0].focus();
      } else {
        container.setAttribute('tabIndex', '-1');
        container.focus();
      }
    }

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (onClose) {
          e.stopPropagation();
          onClose();
        }
        return;
      }

      if (e.key === 'Tab') {
        const currentContainer = containerRef?.current;
        if (!currentContainer) return;

        const focusables = Array.from(
          currentContainer.querySelectorAll(FOCUSABLE_SELECTOR)
        ).filter(el => {
          if (el.getAttribute('aria-hidden') === 'true') return false;
          const style = window.getComputedStyle(el);
          return style.display !== 'none' && style.visibility !== 'hidden';
        });

        if (focusables.length === 0) {
          e.preventDefault();
          return;
        }

        const firstElement = focusables[0];
        const lastElement = focusables[focusables.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === firstElement || !currentContainer.contains(document.activeElement)) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement || !currentContainer.contains(document.activeElement)) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      if (previousFocusRef.current && typeof previousFocusRef.current.focus === 'function') {
        previousFocusRef.current.focus();
      }
    };
  }, [isOpen, onClose, containerRef]);
}

export default useFocusTrap;
