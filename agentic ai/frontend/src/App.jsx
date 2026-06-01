import { useState, useCallback, useEffect } from 'react';
import { Zap, Lightbulb, X, Layers } from 'lucide-react';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import AdminPage from './pages/AdminPage';
import { postChat } from './services/api';
import './index.css';

/* ── Suggestion Queries ─────────────────────────────────────────── */

const PINNED_QUERIES = [
  'How do I connect an LED to Arduino pin 13?',
  'What is the ATmega328P and what are its specifications?',
  'Show me the GPIO pinout for Raspberry Pi and explain pin numbering',
  'It is not working',
  'What resistor value should I use for that LED?',
];

const EXTRA_QUERIES = [
  'Show Arduino Nano pinout and pin functions',
  'What are the SPI communication pins for Arduino Uno?',
  'How to wire up a light sensor or photoresistor?',
  'Where is the debug connector SWD located on Raspberry Pi Pico W?',
  'What are the key wireless features and speed of the RM2 radio module?',
  'How do I mount and program a bootloader on ATmega328P?',
  'How to establish a secure SSH connection to a Raspberry Pi?',
];

/* ── Helpers ────────────────────────────────────────────────────── */

function getSessionId() {
  let id = sessionStorage.getItem('session_id');
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem('session_id', id);
  }
  return id;
}

/* ── App ────────────────────────────────────────────────────────── */

export default function App() {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(getSessionId);
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('Analyzing query...');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  // Sync state with browser location (back / forward clicks)
  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const handleSend = useCallback(
    async (query, imageFile = null, globalSearchRequested = false) => {
      if (!globalSearchRequested) {
        const userMsg = {
          id: crypto.randomUUID(),
          sender: 'user',
          text: query,
          inputImageUrl: imageFile ? URL.createObjectURL(imageFile) : null,
        };
        setMessages((prev) => [...prev, userMsg]);
      }
      setLoading(true);
      setLoadingText('Processing request...');
      setShowSuggestions(false);

      try {
        const formData = new FormData();
        formData.append('query', query || '');
        formData.append('session_id', sessionId || '');
        formData.append('global_search_requested', globalSearchRequested ? 'true' : 'false');
        formData.append('stream', 'true');
        if (imageFile) {
          formData.append('image', imageFile);
        }

        const response = await fetch('/chat', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop(); // Keep remaining incomplete line

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              const payload = JSON.parse(trimmed.substring(6));
              if (payload.type === 'status') {
                setLoadingText(payload.text);
              } else if (payload.type === 'result') {
                const res = payload.data;
                if (res.session_id && res.session_id !== sessionId) {
                  setSessionId(res.session_id);
                  sessionStorage.setItem('session_id', res.session_id);
                }

                const assistantMsg = {
                  id: crypto.randomUUID(),
                  sender: 'assistant',
                  text: res.answer || '',
                  userQuery: query,
                  imageUrl: res.image_url || null,
                  imageCaption: res.image_caption || null,
                  sources: res.sources || [],
                  intent: res.intent || null,
                  cacheHit: res.cache_hit || false,
                  coverageFound: res.coverage_found ?? true,
                  offerGlobalSearch: res.offer_global_search ?? false,
                  globalSearchRequested: res.global_search_requested ?? false,
                };
                setMessages((prev) => [...prev, assistantMsg]);
              } else if (payload.type === 'error') {
                throw new Error(payload.detail || 'Stream processing failed');
              }
            }
          }
        }
      } catch (err) {
        const errorMsg = {
          id: crypto.randomUUID(),
          sender: 'assistant',
          text: `Error: ${err.message}`,
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId]
  );

  const handleReset = () => {
    const newId = crypto.randomUUID();
    sessionStorage.setItem('session_id', newId);
    setSessionId(newId);
    setMessages([]);
  };

  const handleBackToChat = () => {
    window.history.pushState({}, '', '/');
    setCurrentPath('/');
  };

  const handleNavigateToAdmin = () => {
    window.history.pushState({}, '', '/admin');
    setCurrentPath('/admin');
  };

  const hasMessages = messages.length > 0;

  // Render Admin View if Route matches
  if (currentPath === '/admin') {
    return <AdminPage onBackToChat={handleBackToChat} />;
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <Zap size={20} className="header-icon" />
          <h1 className="header-title">Electronics Assistant</h1>
        </div>
        <div className="header-right">
          <button
            className="icon-button"
            onClick={handleNavigateToAdmin}
            aria-label="Manage Knowledge Base"
            title="Knowledge Base Admin"
          >
            <Layers size={18} />
          </button>
          <button
            className="icon-button"
            onClick={() => setShowSuggestions((v) => !v)}
            aria-label="Toggle suggestions"
            title="Suggestions"
          >
            <Lightbulb size={18} />
          </button>
          {hasMessages && (
            <button
              className="icon-button"
              onClick={handleReset}
              aria-label="New chat"
              title="New chat"
            >
              <X size={18} />
            </button>
          )}
        </div>
      </header>

      {/* Suggestions dropdown */}
      {showSuggestions && (
        <div className="suggestions-dropdown">
          <div className="suggestions-section">
            <span className="suggestions-heading">Demo Queries</span>
            {PINNED_QUERIES.map((q, i) => (
              <button
                key={`pin-${i}`}
                className="suggestion-chip"
                onClick={() => handleSend(q, null)}
              >
                {q}
              </button>
            ))}
          </div>
          <div className="suggestions-section">
            <span className="suggestions-heading">More Examples</span>
            {EXTRA_QUERIES.map((q, i) => (
              <button
                key={`ext-${i}`}
                className="suggestion-chip"
                onClick={() => handleSend(q, null)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="app-main">
        {!hasMessages && !loading ? (
          <div className="welcome">
            <Zap size={40} className="welcome-icon" />
            <h2 className="welcome-title">
              Multimodal Electronics Assistant
            </h2>
            <p className="welcome-subtitle">
              Ask about Arduino, Raspberry Pi, pinouts, wiring diagrams, and
              more.
            </p>
            <div className="welcome-chips">
              {PINNED_QUERIES.map((q, i) => (
                <button
                  key={i}
                  className="suggestion-chip"
                  onClick={() => handleSend(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <ChatWindow messages={messages} loading={loading} loadingText={loadingText} onSend={handleSend} />
        )}
      </main>

      {/* Input */}
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  );
}
