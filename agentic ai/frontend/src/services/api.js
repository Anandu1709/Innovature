import axios from 'axios';

const api = axios.create({
  headers: { 'Content-Type': 'application/json' },
});

/**
 * Send a chat query to the backend.
 * @param {string} query - User's question
 * @param {string|null} sessionId - Current session ID
 * @returns {Promise<object>} ChatResponse from backend
 */
export async function postChat(query, sessionId) {
  const { data } = await api.post('/chat', {
    query,
    session_id: sessionId,
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
