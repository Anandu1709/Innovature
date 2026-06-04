import { useState } from 'react';
import { User, Bot, ExternalLink, Zap, Download } from 'lucide-react';

// --- Inline Token Parser for Bold & Citations -------------------------------
function parseInline(text, sources) {
  const citationRegex = /\[Source:\s*([^\]]+)\]/g;
  const boldRegex = /\*\*([^*]+)\*\*/g;
  
  let tokens = [{ type: 'text', content: text }];
  
  // 1. Parse Citations
  let nextTokens = [];
  for (const token of tokens) {
    if (token.type !== 'text') {
      nextTokens.push(token);
      continue;
    }
    
    let lastIdx = 0;
    let match;
    citationRegex.lastIndex = 0;
    while ((match = citationRegex.exec(token.content)) !== null) {
      const before = token.content.substring(lastIdx, match.index);
      if (before) nextTokens.push({ type: 'text', content: before });
      
      const sourceName = match[1].trim();
      const cleanName = (n) => n.replace(/^.*[\\\/]/, '').trim().toLowerCase();
      
      const idx = sources.findIndex(
        s => cleanName(s).includes(cleanName(sourceName)) || cleanName(sourceName).includes(cleanName(s))
      );
      
      if (idx !== -1) {
        nextTokens.push({ type: 'citation', index: idx + 1, source: sources[idx] });
      } else {
        nextTokens.push({ type: 'text', content: match[0] });
      }
      lastIdx = citationRegex.lastIndex;
    }
    const after = token.content.substring(lastIdx);
    if (after) nextTokens.push({ type: 'text', content: after });
  }
  tokens = nextTokens;
  
  // 2. Parse Bold Text
  nextTokens = [];
  for (const token of tokens) {
    if (token.type !== 'text') {
      nextTokens.push(token);
      continue;
    }
    
    let lastIdx = 0;
    let match;
    boldRegex.lastIndex = 0;
    while ((match = boldRegex.exec(token.content)) !== null) {
      const before = token.content.substring(lastIdx, match.index);
      if (before) nextTokens.push({ type: 'text', content: before });
      
      nextTokens.push({ type: 'bold', content: match[1] });
      lastIdx = boldRegex.lastIndex;
    }
    const after = token.content.substring(lastIdx);
    if (after) nextTokens.push({ type: 'text', content: after });
  }
  tokens = nextTokens;
  
  return tokens;
}

// --- Render parsed tokens to React components -------------------------------
function renderTokens(tokens, onCitationClick, activeCitation) {
  return tokens.map((token, i) => {
    if (token.type === 'text') {
      return token.content;
    }
    if (token.type === 'bold') {
      return <strong key={i}>{token.content}</strong>;
    }
    if (token.type === 'citation') {
      const isActive = activeCitation === token.index;
      return (
        <span
          key={i}
          className={`citation-sup ${isActive ? 'active' : ''}`}
          onClick={() => onCitationClick(token.index)}
          title={`Source ${token.index}: ${token.source}`}
        >
          {token.index}
        </span>
      );
    }
    return null;
  });
}

// --- Parse and render structured Markdown blocks (headings, lists, paras) ---
function renderStructuredText(text, sources, onCitationClick, activeCitation) {
  if (!text) return null;
  
  const lines = text.split('\n');
  const blocks = [];
  let currentList = null;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    
    if (!trimmed) {
      if (currentList) {
        blocks.push(currentList);
        currentList = null;
      }
      continue;
    }
    
    // Check for headers (e.g. ### Header or ## Header or 1. **Header**)
    if (trimmed.startsWith('### ') || trimmed.startsWith('## ')) {
      if (currentList) {
        blocks.push(currentList);
        currentList = null;
      }
      const headingText = trimmed.replace(/^#+\s+/, '');
      const tokens = parseInline(headingText, sources);
      blocks.push(
        <h4 key={`h-${i}`} className="message-heading">
          {renderTokens(tokens, onCitationClick, activeCitation)}
        </h4>
      );
      continue;
    }
    
    // Check for bullet lists (* or -)
    const bulletMatch = line.match(/^(\s*)[*-]\s+(.*)$/);
    if (bulletMatch) {
      const itemText = bulletMatch[2];
      const tokens = parseInline(itemText, sources);
      
      if (!currentList || currentList.type !== 'ul') {
        if (currentList) blocks.push(currentList);
        currentList = {
          type: 'ul',
          key: `ul-${i}`,
          items: []
        };
      }
      
      currentList.items.push(
        <li key={`li-${i}`}>
          {renderTokens(tokens, onCitationClick, activeCitation)}
        </li>
      );
      continue;
    }
    
    // Check for numbered lists (e.g. 1. )
    const numberMatch = line.match(/^(\s*)\d+\.\s+(.*)$/);
    if (numberMatch) {
      const itemText = numberMatch[2];
      const tokens = parseInline(itemText, sources);
      
      if (!currentList || currentList.type !== 'ol') {
        if (currentList) blocks.push(currentList);
        currentList = {
          type: 'ol',
          key: `ol-${i}`,
          items: []
        };
      }
      
      currentList.items.push(
        <li key={`li-${i}`}>
          {renderTokens(tokens, onCitationClick, activeCitation)}
        </li>
      );
      continue;
    }
    
    // Standard paragraph
    if (currentList) {
      blocks.push(currentList);
      currentList = null;
    }
    
    const tokens = parseInline(line, sources);
    blocks.push(
      <p key={`p-${i}`} className="message-paragraph">
        {renderTokens(tokens, onCitationClick, activeCitation)}
      </p>
    );
  }
  
  if (currentList) {
    blocks.push(currentList);
  }
  
  return blocks.map((block) => {
    if (block.type === 'ul') {
      return (
        <ul key={block.key} className="message-list-bullet">
          {block.items}
        </ul>
      );
    }
    if (block.type === 'ol') {
      return (
        <ol key={block.key} className="message-list-number">
          {block.items}
        </ol>
      );
    }
    return block;
  });
}

// --- Main MessageBubble Component ------------------------------------------
export default function MessageBubble({ message, onSend }) {
  const isUser = message.sender === 'user';
  const [expanded, setExpanded] = useState(false);
  const [activeCitation, setActiveCitation] = useState(null);
  const [searchCanceled, setSearchCanceled] = useState(false);

  // Helper to extract clean filename (without folders/paths)
  const getShortName = (fullName) => {
    return fullName.split('/').pop().split('\\').pop();
  };

  const handleCitationClick = (index) => {
    setActiveCitation(index);
    setExpanded(true); // Automatically expand sources tag list so they can see it
    setTimeout(() => {
      // Clear highlight after 2.5s for clean feedback
      setActiveCitation(null);
    }, 2500);
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(message.imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const filename = message.imageUrl.split('/').pop() || 'technical-diagram.png';
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download image:', err);
      const a = document.createElement('a');
      a.href = message.imageUrl;
      a.target = '_blank';
      a.download = '';
      a.click();
    }
  };

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

        <div className="message-text">
          {isUser ? (
            message.text
          ) : (
            renderStructuredText(message.text, message.sources || [], handleCitationClick, activeCitation)
          )}
        </div>


        {message.imageUrl && (
          <div className="message-image-container">
            <button
              className="image-download-btn"
              onClick={handleDownload}
              aria-label="Download image"
              title="Download image"
            >
              <Download size={16} />
            </button>
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
            {expanded || message.sources.length <= 1 ? (
              <>
                {message.sources.map((src, i) => {
                  const idx = i + 1;
                  const isActive = activeCitation === idx;
                  return (
                    <span key={i} className={`source-tag ${isActive ? 'active' : ''}`}>
                      <span className="source-index">[{idx}]</span>
                      <ExternalLink size={10} />
                      {getShortName(src)}
                    </span>
                  );
                })}
                {message.sources.length > 1 && (
                  <button
                    onClick={() => setExpanded(false)}
                    className="source-toggle-btn"
                    title="Collapse sources list"
                  >
                    Show less
                  </button>
                )}
              </>
            ) : (
              <>
                <span className={`source-tag ${activeCitation === 1 ? 'active' : ''}`}>
                  <span className="source-index">[1]</span>
                  <ExternalLink size={10} />
                  {getShortName(message.sources[0])}
                </span>
                <button
                  onClick={() => setExpanded(true)}
                  className="source-toggle-btn"
                  title="Expand all sources"
                >
                  +{message.sources.length - 1} more
                </button>
              </>
            )}
          </div>
        )}

        {/* Global Web Search Fallback Offers */}
        {message.offerGlobalSearch && (
          <div className="search-fallback-container">
            {!searchCanceled ? (
              <div className="search-fallback-buttons">
                <button
                  className="search-fallback-btn primary"
                  onClick={() => onSend(message.userQuery, null, true)}
                >
                  Search Web
                </button>
                <button
                  className="search-fallback-btn secondary"
                  onClick={() => setSearchCanceled(true)}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <div className="search-fallback-canceled">
                General search canceled
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
