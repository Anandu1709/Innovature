import { User, Bot, ExternalLink, Zap } from 'lucide-react';

export default function MessageBubble({ message }) {
  const isUser = message.sender === 'user';

  return (
    <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-avatar">
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      <div className="message-content">
        {/* User-uploaded image */}
        {message.inputImageUrl && (
          <div className="message-input-image">
            <img
              src={message.inputImageUrl}
              alt="Uploaded"
              className="input-image-thumb"
            />
          </div>
        )}

        <div className="message-text">{message.text}</div>

        {message.cacheHit && (
          <div className="cache-badge">
            <Zap size={11} />
            <span>Cached</span>
          </div>
        )}

        {message.imageUrl && (
          <div className="message-image-container">
            <img
              src={message.imageUrl}
              alt={message.imageCaption || 'Technical diagram'}
              className="message-image"
              loading="lazy"
            />
            {message.imageCaption && (
              <p className="message-image-caption">{message.imageCaption}</p>
            )}
          </div>
        )}

        {message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <span className="sources-label">Sources:</span>
            {message.sources.map((src, i) => (
              <span key={i} className="source-tag">
                <ExternalLink size={12} />
                {src}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
