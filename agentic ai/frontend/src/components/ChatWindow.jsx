import { useEffect, useRef, useState } from 'react';
import MessageBubble from './MessageBubble';
import { Loader } from 'lucide-react';

// --- Stateful Dynamic Loader Subcomponent (Text Only, No Emojis) ------------
function DynamicLoading({ hasImage }) {
  const [status, setStatus] = useState('');

  useEffect(() => {
    const imageStatuses = [
      { time: 0, text: 'Processing uploaded image and identifying components...' },
      { time: 1800, text: 'Searching documentation database for specifications...' },
      { time: 3800, text: 'Retrieving relevant technical schematics and pinouts...' },
      { time: 5800, text: 'Synthesizing multimodal explanation...' },
      { time: 7800, text: 'Finalizing answer with source citations...' }
    ];

    const textStatuses = [
      { time: 0, text: 'Analyzing user query context...' },
      { time: 1500, text: 'Searching documentation database for specifications...' },
      { time: 3500, text: 'Synthesizing detailed explanation...' },
      { time: 5500, text: 'Finalizing answer with source citations...' }
    ];

    const statuses = hasImage ? imageStatuses : textStatuses;

    // Set initial status
    setStatus(statuses[0].text);

    // Schedule subsequent updates
    const timers = statuses.slice(1).map((s) => {
      return setTimeout(() => {
        setStatus(s.text);
      }, s.time);
    });

    return () => {
      timers.forEach(clearTimeout);
    };
  }, [hasImage]);

  return (
    <div className="loading-indicator">
      <Loader size={16} className="spinner" />
      {/* Key-force remount ensures CSS fade-in animation triggers on every change */}
      <span key={status} className="loading-status-text">
        {status}
      </span>
    </div>
  );
}

export default function ChatWindow({ messages, loading, onSend }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Check if the current pending query contains an image
  const lastMsg = messages[messages.length - 1];
  const hasImage = lastMsg && lastMsg.sender === 'user' && !!lastMsg.inputImageUrl;

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onSend={onSend} />
        ))}

        {loading && <DynamicLoading hasImage={hasImage} />}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
