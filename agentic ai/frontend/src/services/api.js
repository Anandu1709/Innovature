import axios from 'axios';

const api = axios.create({});

/**
 * Send a chat query to the backend.
 * Supports text-only, image-only, and image+text via multipart/form-data.
 *
 * @param {string} query - User's question (can be empty if image provided)
 * @param {string|null} sessionId - Current session ID
 * @param {File|null} imageFile - Optional image file to upload
 * @returns {Promise<object>} ChatResponse from backend
 */
export async function postChat(query, sessionId, imageFile = null) {
  const formData = new FormData();
  formData.append('query', query || '');
  formData.append('session_id', sessionId || '');

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
