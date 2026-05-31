import React, { useState, useRef } from 'react';
import { Upload, Link2, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { uploadAdminPDF, ingestAdminURL, ingestAdminMarkdown } from '../../services/api';

export default function UploadPanel({ onIngestionTriggered }) {
  const [activeTab, setActiveTab] = useState('pdf');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // PDF Form State
  const [pdfFile, setPdfFile] = useState(null);
  const fileInputRef = useRef(null);

  // URL Form State
  const [urlInput, setUrlInput] = useState('');

  // Markdown Form State
  const [mdTitle, setMdTitle] = useState('');
  const [mdContent, setMdContent] = useState('');

  const clearStatus = () => {
    setError(null);
    setSuccess(null);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    clearStatus();
  };

  // 1. PDF Handlers
  const handleFileDrop = (e) => {
    e.preventDefault();
    clearStatus();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setPdfFile(droppedFile);
    } else {
      setError('Please drop a valid PDF file.');
    }
  };

  const handleFileSelect = (e) => {
    clearStatus();
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setPdfFile(selectedFile);
    }
  };

  const submitPDF = async (e) => {
    e.preventDefault();
    if (!pdfFile) return;

    setLoading(true);
    clearStatus();
    try {
      const res = await uploadAdminPDF(pdfFile);
      setSuccess(res.message || 'PDF uploaded successfully. Processing in background.');
      setPdfFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      if (onIngestionTriggered) onIngestionTriggered();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload PDF document.');
    } finally {
      setLoading(false);
    }
  };

  // 2. URL Handlers
  const submitURL = async (e) => {
    e.preventDefault();
    const urlStr = urlInput.trim();
    if (!urlStr) return;

    setLoading(true);
    clearStatus();
    try {
      const res = await ingestAdminURL(urlStr);
      setSuccess(res.message || 'URL registered successfully. Scraping in background.');
      setUrlInput('');
      if (onIngestionTriggered) onIngestionTriggered();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to index target URL.');
    } finally {
      setLoading(false);
    }
  };

  // 3. Markdown Handlers
  const submitMarkdown = async (e) => {
    e.preventDefault();
    const title = mdTitle.trim();
    const content = mdContent.trim();
    if (!title || !content) return;

    setLoading(true);
    clearStatus();
    try {
      const res = await ingestAdminMarkdown(title, content);
      setSuccess(res.message || 'Markdown notes registered. Chunking in background.');
      setMdTitle('');
      setMdContent('');
      if (onIngestionTriggered) onIngestionTriggered();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to submit Markdown block.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="admin-upload-panel">
      {/* Tab Navigation */}
      <div className="upload-tabs-header">
        <button
          className={`upload-tab-btn ${activeTab === 'pdf' ? 'active' : ''}`}
          onClick={() => handleTabChange('pdf')}
        >
          <Upload size={14} />
          <span>Upload PDF</span>
        </button>
        <button
          className={`upload-tab-btn ${activeTab === 'url' ? 'active' : ''}`}
          onClick={() => handleTabChange('url')}
        >
          <Link2 size={14} />
          <span>Documentation URL</span>
        </button>
        <button
          className={`upload-tab-btn ${activeTab === 'markdown' ? 'active' : ''}`}
          onClick={() => handleTabChange('markdown')}
        >
          <FileText size={14} />
          <span>Pasted Notes</span>
        </button>
      </div>

      {/* Tab Content Box */}
      <div className="upload-tabs-content">
        {/* Status Alerts */}
        {error && (
          <div className="status-alert error">
            <AlertCircle size={15} />
            <span>{error}</span>
          </div>
        )}
        {success && (
          <div className="status-alert success">
            <CheckCircle size={15} />
            <span>{success}</span>
          </div>
        )}

        {/* 1. PDF Tab Content */}
        {activeTab === 'pdf' && (
          <form onSubmit={submitPDF} className="tab-form flex-col">
            <div
              className="dropzone-area"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={28} className="dropzone-icon" />
              {pdfFile ? (
                <span className="dropzone-text highlight">{pdfFile.name}</span>
              ) : (
                <span className="dropzone-text">
                  Drag & drop datasheet PDF here, or <span className="underline-link">browse files</span>
                </span>
              )}
              <span className="dropzone-hint">Max file size 25MB (PDF only)</span>
            </div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".pdf"
              style={{ display: 'none' }}
            />
            <button
              type="submit"
              className="admin-action-btn primary"
              disabled={loading || !pdfFile}
            >
              {loading ? 'Processing Upload...' : 'Ingest PDF'}
            </button>
          </form>
        )}

        {/* 2. URL Tab Content */}
        {activeTab === 'url' && (
          <form onSubmit={submitURL} className="tab-form flex-col">
            <div className="input-group">
              <label htmlFor="url-input" className="input-label">
                Documentation URL
              </label>
              <input
                id="url-input"
                type="url"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://docs.arduino.cc/learn/starting-guide/getting-started-arduino/"
                className="admin-text-input"
                required
              />
              <span className="field-hint">
                Backend will fetch the web content, extract structural elements, download diagram resources, and generate embeddings.
              </span>
            </div>
            <button
              type="submit"
              className="admin-action-btn primary"
              disabled={loading || !urlInput.trim()}
            >
              {loading ? 'Scraping Page...' : 'Fetch & Index URL'}
            </button>
          </form>
        )}

        {/* 3. Markdown Tab Content */}
        {activeTab === 'markdown' && (
          <form onSubmit={submitMarkdown} className="tab-form flex-col">
            <div className="input-group">
              <label htmlFor="md-title" className="input-label">
                Title / Identifier
              </label>
              <input
                id="md-title"
                type="text"
                value={mdTitle}
                onChange={(e) => setMdTitle(e.target.value)}
                placeholder="e.g. Pinout Configuration Notes"
                className="admin-text-input"
                required
              />
            </div>
            <div className="input-group">
              <label htmlFor="md-content" className="input-label">
                Markdown / Text Content
              </label>
              <textarea
                id="md-content"
                value={mdContent}
                onChange={(e) => setMdContent(e.target.value)}
                placeholder="# Hardware Interface Guide&#10;&#10;Explain pinouts, configurations, or datasheet properties here..."
                rows={6}
                className="admin-textarea"
                required
              />
            </div>
            <button
              type="submit"
              className="admin-action-btn primary"
              disabled={loading || !mdTitle.trim() || !mdContent.trim()}
            >
              {loading ? 'Ingesting Chunks...' : 'Save & Index Notes'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
