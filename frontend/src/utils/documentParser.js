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
 * Extracts plain text from a PDF File or Blob.
 */
export async function extractTextFromPDF(fileOrBlob) {
  try {
    const arrayBuffer = await fileOrBlob.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(arrayBuffer) });
    const pdf = await loadingTask.promise;
    let fullText = '';

    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map(item => ('str' in item ? item.str : ''))
        .join(' ');
      
      if (pageText.trim()) {
        fullText += `--- [Pagina ${pageNum}] ---\n${pageText.trim()}\n\n`;
      }
    }

    return fullText.trim() || '[Documento PDF vuoto o composto solo da immagini scansite]';
  } catch (err) {
    console.error('Error parsing PDF:', err);
    return `[Errore durante l'estrazione del testo dal PDF: ${err.message || err}]`;
  }
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
export async function extractDocumentText(file) {
  const name = file.name.toLowerCase();

  if (name.endsWith('.pdf') || file.type === 'application/pdf') {
    return await extractTextFromPDF(file);
  }

  if (name.endsWith('.docx') || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    return await extractTextFromDOCX(file);
  }

  // Text, Markdown, Code, JSON, CSV, YAML, etc.
  try {
    return await file.text();
  } catch (err) {
    console.error('Error reading text file:', err);
    return `[Impossibile leggere il file ${file.name}: ${err.message || err}]`;
  }
}
