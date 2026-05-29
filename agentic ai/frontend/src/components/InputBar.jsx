import { useState, useRef } from 'react';
import { SendHorizontal, ImagePlus, X } from 'lucide-react';

const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5 MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export default function InputBar({ onSend, disabled }) {
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate type
    if (!ALLOWED_TYPES.includes(file.type)) {
      alert('Unsupported format. Please use JPG, PNG, or WEBP.');
      return;
    }

    // Validate size
    if (file.size > MAX_IMAGE_SIZE) {
      alert('Image too large. Maximum size is 5 MB.');
      return;
    }

    setImage(file);
    setImagePreview(URL.createObjectURL(file));

    // Reset file input so re-selecting the same file triggers onChange
    e.target.value = '';
  };

  const handleRemoveImage = () => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setImage(null);
    setImagePreview(null);
  };

  const handleSubmit = () => {
    const trimmed = text.trim();
    if ((!trimmed && !image) || disabled) return;

    onSend(trimmed, image);
    setText('');
    handleRemoveImage();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-bar-wrapper">
      {/* Image preview */}
      {imagePreview && (
        <div className="image-preview-bar">
          <div className="image-preview-thumbnail">
            <img src={imagePreview} alt="Upload preview" />
            <button
              className="image-preview-remove"
              onClick={handleRemoveImage}
              aria-label="Remove image"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      <div className="input-bar">
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleImageSelect}
          style={{ display: 'none' }}
        />

        {/* Image upload button */}
        <button
          className="image-upload-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          aria-label="Upload image"
          title="Upload image (JPG, PNG, WEBP — max 5MB)"
        >
          <ImagePlus size={18} />
        </button>

        <input
          type="text"
          className="input-field"
          placeholder={image ? 'Add a question about this image...' : 'Ask about Arduino, Raspberry Pi...'}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          autoFocus
        />
        <button
          className="send-button"
          onClick={handleSubmit}
          disabled={disabled || (!text.trim() && !image)}
          aria-label="Send message"
        >
          <SendHorizontal size={18} />
        </button>
      </div>
    </div>
  );
}
