import { useEffect, useRef, useState } from 'react';
import MessageBubble from './MessageBubble';
import { Loader } from 'lucide-react';

// --- Stateful Dynamic Loader Subcomponent (Text Only, No Emojis) ------------
function DynamicLoading({ text }) {
  return (
    <div className="loading-indicator">
      <Loader size={16} className="spinner" />
      {/* Key-force remount ensures CSS fade-in animation triggers on every change */}
      <span key={text} className="loading-status-text">
        {text || 'Processing query...'}
      </span>
    </div>
  );
}

export default function ChatWindow({ messages, loading, loadingText, onSend }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onSend={onSend} />
        ))}

        {loading && <DynamicLoading text={loadingText} />}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
