import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import { Loader } from 'lucide-react';

export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {loading && (
          <div className="loading-indicator">
            <Loader size={20} className="spinner" />
            <span>Thinking...</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
