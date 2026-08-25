import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import mammoth from 'mammoth';

// Configure pdf.js worker for client-side bundle
try {
  pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/legacy/build/pdf.worker.min.mjs',
    import.meta.url
  ).toString();
} catch {
  // fallback if worker URL fails
}

/**
 * Parses user page range strings such as "1-3, 5", "1,2,4", "1-10".
 * Returns a sorted unique array of 1-indexed page numbers clamped to [1, maxPages].
 */
export function parsePageRange(rangeStr, maxPages) {
  if (!rangeStr || !rangeStr.trim() || maxPages <= 0) {
    return Array.from({ length: Math.max(1, maxPages) }, (_, i) => i + 1);
  }

  const pages = new Set();
  const parts = rangeStr.split(/[,;\s]+/);

  for (const part of parts) {
    const trimmed = part.trim();
    if (!trimmed) continue;

    if (trimmed.includes('-')) {
      const [startStr, endStr] = trimmed.split('-');
      const start = parseInt(startStr, 10);
      const end = parseInt(endStr, 10);
      if (!isNaN(start) && !isNaN(end)) {
        const from = Math.max(1, Math.min(start, end));
        const to = Math.min(maxPages, Math.max(start, end));
        for (let p = from; p <= to; p++) {
          pages.add(p);
        }
      }
    } else {
      const p = parseInt(trimmed, 10);
      if (!isNaN(p) && p >= 1 && p <= maxPages) {
        pages.add(p);
      }
    }
  }

  const sorted = Array.from(pages).sort((a, b) => a - b);
  return sorted.length > 0 ? sorted : Array.from({ length: maxPages }, (_, i) => i + 1);
}

/**
 * Estimates token count for text (~4 chars per token for European languages).
 */
export function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(text.trim().length / 3.8);
}

/**
 * Cleans extracted document text by removing repeated headers, footers,
 * disclaimers across pages, and collapsing excessive whitespace.
 */
export function cleanDocumentPages(pages) {
  if (!pages || pages.length === 0) return [];
  if (pages.length === 1) {
    const cleaned = cleanSingleText(pages[0].text);
    return [{ ...pages[0], text: cleaned }];
  }

  // 1. Collect lines frequency across pages to identify repetitive headers/footers
  const lineCountMap = new Map();
  pages.forEach(p => {
    const lines = (p.text || '').split('\n').map(l => l.trim()).filter(l => l.length > 5);
    const uniqueInPage = new Set(lines);
    uniqueInPage.forEach(line => {
      lineCountMap.set(line, (lineCountMap.get(line) || 0) + 1);
    });
  });

  // Lines that appear in more than 50% of pages (or >= 3 pages) are considered boilerplate
  const boilerplateThreshold = Math.max(2, Math.floor(pages.length * 0.5));
  const boilerplateLines = new Set();
  lineCountMap.forEach((count, line) => {
    if (count >= boilerplateThreshold) {
      // Check if line looks like boilerplate (disclaimer, footer, website, copyright)
      if (
        line.match(/(tutti i diritti|all rights reserved|copyright|privacy policy|pag\.|pagina \d+|www\.|http|toyota motor|disclaimer)/i) ||
        line.length < 80
      ) {
        boilerplateLines.add(line);
      }
    }
  });

  return pages.map(p => {
    const lines = (p.text || '').split('\n');
    const filteredLines = lines.filter(l => {
      const trimmed = l.trim();
      if (!trimmed) return false;
      if (boilerplateLines.has(trimmed)) return false;
      // Filter out standalone page numbers
      if (trimmed.match(/^(?:pag(?:ina)?\.?\s*\d+(?:\s*(?:\/|di)\s*\d+)?|\d+\s*\/\s*\d+|\d+)$/i)) return false;
      return true;
    });

    const cleanedText = cleanSingleText(filteredLines.join('\n'));
    return {
      ...p,
      text: cleanedText
    };
  });
}

function cleanSingleText(text) {
  if (!text) return '';
  return text
    .replace(/[ \t]+/g, ' ') // Collapse multiple spaces
    .replace(/\n{3,}/g, '\n\n') // Collapse excessive newlines
    .replace(/^[ \t]+|[ \t]+$/gm, '') // Trim lines
    .trim();
}

/**
 * Extracts plain text pages from a PDF File or Blob.
 */
export async function extractPdfPages(fileOrBlob) {
  try {
    const arrayBuffer = await fileOrBlob.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(arrayBuffer) });
    const pdf = await loadingTask.promise;
    const pages = [];

    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map(item => ('str' in item ? item.str : ''))
        .join(' ');
      
      pages.push({
        pageNum,
        text: pageText.trim()
      });
    }

    return {
      numPages: pdf.numPages,
      pages
    };
  } catch (err) {
    console.error('Error parsing PDF pages:', err);
    return {
      numPages: 0,
      pages: [],
      error: err.message || String(err)
    };
  }
}

/**
 * Renders document text according to page selection and cleaning options.
 */
export function formatPdfDocumentText(pages, options = {}) {
  const {
    pageSelection = 'all', // 'all', 'first3', 'custom'
    customRange = '',
    cleanText = true
  } = options;

  if (!pages || pages.length === 0) return '';

  const totalPages = pages.length;
  let targetPageNums = [];

  if (pageSelection === 'first3') {
    targetPageNums = Array.from({ length: Math.min(3, totalPages) }, (_, i) => i + 1);
  } else if (pageSelection === 'custom' && customRange) {
    targetPageNums = parsePageRange(customRange, totalPages);
  } else {
    targetPageNums = Array.from({ length: totalPages }, (_, i) => i + 1);
  }

  const selectedPages = pages.filter(p => targetPageNums.includes(p.pageNum));
  const processedPages = cleanText ? cleanDocumentPages(selectedPages) : selectedPages;

  let fullText = '';
  for (const p of processedPages) {
    if (p.text && p.text.trim()) {
      fullText += `--- [Pagina ${p.pageNum}] ---\n${p.text.trim()}\n\n`;
    }
  }

  return fullText.trim();
}

/**
 * Extracts plain text from a DOCX File or Blob.
 */
export async function extractTextFromDOCX(fileOrBlob) {
  try {
    const arrayBuffer = await fileOrBlob.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value.trim() || '[Documento DOCX vuoto]';
  } catch (err) {
    console.error('Error parsing DOCX:', err);
    return `[Errore durante l'estrazione del testo dal file Word: ${err.message || err}]`;
  }
}

/**
 * Universal document text extractor supporting PDF, DOCX, and all text/code files.
 */
export async function extractDocumentText(file, options = {}) {
  const name = file.name.toLowerCase();

  if (name.endsWith('.pdf') || file.type === 'application/pdf') {
    const pdfData = await extractPdfPages(file);
    if (pdfData.error) {
      return `[Errore durante l'estrazione del PDF: ${pdfData.error}]`;
    }
    return formatPdfDocumentText(pdfData.pages, options);
  }

  if (name.endsWith('.docx') || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    const docxText = await extractTextFromDOCX(file);
    return options.cleanText !== false ? cleanSingleText(docxText) : docxText;
  }

  // Text, Markdown, Code, JSON, CSV, YAML, etc.
  try {
    const text = await file.text();
    return options.cleanText !== false ? cleanSingleText(text) : text;
  } catch (err) {
    console.error('Error reading text file:', err);
    return `[Impossibile leggere il file ${file.name}: ${err.message || err}]`;
  }
}
