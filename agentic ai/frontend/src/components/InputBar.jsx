import { useState } from 'react';
import { SendHorizontal } from 'lucide-react';

export default function InputBar({ onSend, disabled }) {
  const [text, setText] = useState('');

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-bar">
      <input
        type="text"
        className="input-field"
        placeholder="Ask about Arduino, Raspberry Pi..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        autoFocus
      />
      <button
        className="send-button"
        onClick={handleSubmit}
        disabled={disabled || !text.trim()}
        aria-label="Send message"
      >
        <SendHorizontal size={18} />
      </button>
    </div>
  );
}
