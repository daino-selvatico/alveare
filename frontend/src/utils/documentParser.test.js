import { describe, it, expect } from 'vitest';
import {
  parsePageRange,
  estimateTokens,
  cleanDocumentPages,
  formatPdfDocumentText
} from './documentParser';

describe('documentParser utility', () => {
  describe('parsePageRange', () => {
    it('returns all pages when range is empty or invalid', () => {
      expect(parsePageRange('', 5)).toEqual([1, 2, 3, 4, 5]);
      expect(parsePageRange(null, 3)).toEqual([1, 2, 3]);
    });

    it('parses single pages and ranges correctly', () => {
      expect(parsePageRange('1-3, 5', 10)).toEqual([1, 2, 3, 5]);
      expect(parsePageRange('2, 4, 6-8', 10)).toEqual([2, 4, 6, 7, 8]);
    });

    it('clamps page numbers to [1, maxPages]', () => {
      expect(parsePageRange('0-4, 8-15', 10)).toEqual([1, 2, 3, 4, 8, 9, 10]);
    });
  });

  describe('estimateTokens', () => {
    it('estimates tokens proportionally to character length', () => {
      expect(estimateTokens('')).toBe(0);
      expect(estimateTokens('Hello world')).toBeGreaterThan(0);
      expect(estimateTokens('A'.repeat(400))).toBe(106);
    });
  });

  describe('cleanDocumentPages', () => {
    it('filters out boilerplate lines appearing on multiple pages', () => {
      const mockPages = [
        {
          pageNum: 1,
          text: 'Toyota Yaris Hybrid\nPrezzo: 24.000€\nCopyright 2026 Toyota Motor Italia. Tutti i diritti riservati.\nPagina 1 / 3'
        },
        {
          pageNum: 2,
          text: 'Optional: Cerchi in lega 17"\nColore: Bi-tone metallizzato\nCopyright 2026 Toyota Motor Italia. Tutti i diritti riservati.\nPagina 2 / 3'
        },
        {
          pageNum: 3,
          text: 'Garanzia 5 anni\nAssistenza inclusa\nCopyright 2026 Toyota Motor Italia. Tutti i diritti riservati.\nPagina 3 / 3'
        }
      ];

      const cleaned = cleanDocumentPages(mockPages);
      expect(cleaned[0].text).toContain('Toyota Yaris Hybrid');
      expect(cleaned[0].text).toContain('Prezzo: 24.000€');
      expect(cleaned[0].text).not.toContain('Copyright 2026 Toyota Motor Italia');
      expect(cleaned[0].text).not.toContain('Pagina 1 / 3');

      expect(cleaned[1].text).toContain('Optional: Cerchi in lega 17"');
      expect(cleaned[1].text).not.toContain('Copyright 2026 Toyota Motor Italia');
    });
  });

  describe('formatPdfDocumentText', () => {
    const mockPages = [
      { pageNum: 1, text: 'Pagina uno con dettagli' },
      { pageNum: 2, text: 'Pagina due con optional' },
      { pageNum: 3, text: 'Pagina tre con prezzi' },
      { pageNum: 4, text: 'Pagina quattro con note' }
    ];

    it('formats all pages when pageSelection is all', () => {
      const result = formatPdfDocumentText(mockPages, { pageSelection: 'all', cleanText: false });
      expect(result).toContain('--- [Pagina 1] ---');
      expect(result).toContain('--- [Pagina 4] ---');
    });

    it('formats first 3 pages when pageSelection is first3', () => {
      const result = formatPdfDocumentText(mockPages, { pageSelection: 'first3', cleanText: false });
      expect(result).toContain('--- [Pagina 1] ---');
      expect(result).toContain('--- [Pagina 3] ---');
      expect(result).not.toContain('--- [Pagina 4] ---');
    });

    it('formats custom range when pageSelection is custom', () => {
      const result = formatPdfDocumentText(mockPages, { pageSelection: 'custom', customRange: '1, 4', cleanText: false });
      expect(result).toContain('--- [Pagina 1] ---');
      expect(result).not.toContain('--- [Pagina 2] ---');
      expect(result).toContain('--- [Pagina 4] ---');
    });
  });
});
