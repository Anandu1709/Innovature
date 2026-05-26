import { useState, useCallback } from 'react';
import { Zap, Lightbulb, X } from 'lucide-react';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
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
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleSend = useCallback(
    async (query) => {
      const userMsg = {
        id: crypto.randomUUID(),
        sender: 'user',
        text: query,
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      setShowSuggestions(false);

      try {
        const res = await postChat(query, sessionId);

        // Update session ID if server returned a new one
        if (res.session_id && res.session_id !== sessionId) {
          setSessionId(res.session_id);
          sessionStorage.setItem('session_id', res.session_id);
        }

        const assistantMsg = {
          id: crypto.randomUUID(),
          sender: 'assistant',
          text: res.answer || '',
          imageUrl: res.image_url || null,
          imageCaption: res.image_caption || null,
          sources: res.sources || [],
          intent: res.intent || null,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const errorMsg = {
          id: crypto.randomUUID(),
          sender: 'assistant',
          text: `Error: ${err.response?.data?.detail || err.message}`,
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

  const hasMessages = messages.length > 0;

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
                onClick={() => handleSend(q)}
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
                onClick={() => handleSend(q)}
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
          <ChatWindow messages={messages} loading={loading} />
        )}
      </main>

      {/* Input */}
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  );
}
