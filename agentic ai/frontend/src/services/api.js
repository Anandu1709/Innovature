import axios from 'axios';

const api = axios.create({});

// Automatically attach admin secret key if present in localStorage
api.interceptors.request.use(
  (config) => {
    const secret = localStorage.getItem('admin_secret_key');
    if (secret) {
      config.headers['X-Admin-Secret-Key'] = secret;
    }
    return config;
  },
  (err) => Promise.reject(err)
);

/**
 * Send a chat query to the backend.
 * Supports text-only, image-only, and image+text via multipart/form-data.
 *
 * @param {string} query - User's question (can be empty if image provided)
 * @param {string|null} sessionId - Current session ID
 * @param {File|null} imageFile - Optional image file to upload
 * @returns {Promise<object>} ChatResponse from backend
 */
export async function postChat(query, sessionId, imageFile = null, globalSearchRequested = false) {
  const formData = new FormData();
  formData.append('query', query || '');
  formData.append('session_id', sessionId || '');
  formData.append('global_search_requested', globalSearchRequested ? 'true' : 'false');

  if (imageFile) {
    formData.append('image', imageFile);
  }

  // Let browser set Content-Type with multipart boundary automatically
  const { data } = await api.post('/chat', formData, {
    headers: { 'Content-Type': undefined },
  });
  return data;
}

/**
 * Check backend health status.
 * @returns {Promise<object>} { status, milvus, gemini }
 */
export async function getHealth() {
  const { data } = await api.get('/health');
  return data;
}

/* ── Admin Knowledge Management APIs ────────────────────────────── */

/**
 * Fetch all registered documents.
 * @returns {Promise<Array>} List of registered document records
 */
export async function getAdminDocuments() {
  const { data } = await api.get('/api/admin/documents');
  return data;
}

/**
 * Fetch aggregate metrics (counts for docs, chunks, images).
 * @returns {Promise<object>} Database statistics overview
 */
export async function getAdminStatistics() {
  const { data } = await api.get('/api/admin/statistics');
  return data;
}

/**
 * Upload a PDF file to the ingestion backend.
 * @param {File} pdfFile - PDF File to parse & index
 * @returns {Promise<object>} Background job creation response
 */
export async function uploadAdminPDF(pdfFile) {
  const formData = new FormData();
  formData.append('file', pdfFile);

  const { data } = await api.post('/api/admin/upload/pdf', formData, {
    headers: { 'Content-Type': undefined },
  });
  return data;
}

/**
 * Submit an online documentation URL to the scrap-ingestion service.
 * @param {string} url - Target URL to parse
 * @returns {Promise<object>} Ingestion start response
 */
export async function ingestAdminURL(url) {
  const { data } = await api.post('/api/admin/ingest/url', { url });
  return data;
}

/**
 * Pastes a raw Markdown block to be semantically chunked and indexed.
 * @param {string} title - Identifier for the pasted content
 * @param {string} markdown - Markdown text body
 * @returns {Promise<object>} Ingestion start response
 */
export async function ingestAdminMarkdown(title, markdown) {
  const { data } = await api.post('/api/admin/ingest/markdown', { title, markdown });
  return data;
}

/**
 * Triggers document registry, file chunk, and vector purging from systems.
 * @param {string} documentId - Target registry document ID to remove
 * @returns {Promise<object>} Deletion task response
 */
export async function deleteAdminDocument(documentId) {
  const { data } = await api.delete(`/api/admin/documents/${documentId}`);
  return data;
}
