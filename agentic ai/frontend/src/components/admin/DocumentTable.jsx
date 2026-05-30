import React from 'react';
import { Trash2, File, Link2, FileText, Loader2 } from 'lucide-react';
import { deleteAdminDocument } from '../../services/api';

export default function DocumentTable({ documents, loading, onUpdate }) {
  const getDocTypeIcon = (type) => {
    switch (type) {
      case 'pdf':
        return <File size={15} className="type-icon pdf" />;
      case 'url':
        return <Link2 size={15} className="type-icon url" />;
      case 'markdown':
        return <FileText size={15} className="type-icon md" />;
      default:
        return <File size={15} />;
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'ready':
        return <span className="status-badge ready">Ready</span>;
      case 'processing':
        return (
          <span className="status-badge processing pulsing">
            Processing
          </span>
        );
      case 'deleting':
        return (
          <span className="status-badge deleting">
            Purging...
          </span>
        );
      case 'failed':
        return <span className="status-badge failed">Failed</span>;
      default:
        return <span className="status-badge">{status}</span>;
    }
  };

  const formatDate = (isoString) => {
    if (!isoString) return '-';
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch (e) {
      return isoString;
    }
  };

  const handleDelete = async (id, name) => {
    const confirmPurge = window.confirm(
      `Are you absolutely sure you want to delete "${name}"?\nThis will permanently remove all associated semantic vectors and chunks from both Milvus and BM25 index systems.`
    );
    if (!confirmPurge) return;

    try {
      await deleteAdminDocument(id);
      if (onUpdate) onUpdate();
    } catch (err) {
      alert(`Purge command failed: ${err.response?.data?.detail || 'Network error'}`);
    }
  };

  if (loading) {
    return (
      <div className="table-loader-box">
        <Loader2 className="spinner-icon text-green-accent" size={24} />
        <span>Syncing Knowledge Registry...</span>
      </div>
    );
  }

  return (
    <div className="admin-table-container">
      {documents.length === 0 ? (
        <div className="empty-registry-prompt">
          <span>No documents currently registered. Paste documentation notes or upload a datasheet PDF above to expand retrieval capabilities.</span>
        </div>
      ) : (
        <table className="admin-table">
          <thead>
            <tr>
              <th>Document Name</th>
              <th>Source Type</th>
              <th className="align-center">Semantic Chunks</th>
              <th className="align-center">Diagrams</th>
              <th>Uploaded At</th>
              <th>Ingress Status</th>
              <th className="align-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.document_id}>
                <td className="doc-name-cell" title={doc.name}>
                  {doc.name}
                </td>
                <td className="doc-type-cell">
                  <div className="type-badge">
                    {getDocTypeIcon(doc.type)}
                    <span className="capitalize">{doc.type}</span>
                  </div>
                </td>
                <td className="align-center font-mono">
                  {doc.chunk_count.toLocaleString()}
                </td>
                <td className="align-center font-mono">
                  {doc.image_count.toLocaleString()}
                </td>
                <td className="date-cell">{formatDate(doc.uploaded_at)}</td>
                <td className="status-cell">{getStatusBadge(doc.status)}</td>
                <td className="actions-cell align-center">
                  <button
                    onClick={() => handleDelete(doc.document_id, doc.name)}
                    disabled={doc.status === 'deleting' || doc.status === 'processing'}
                    className="delete-row-btn"
                    title="Delete document and remove from Milvus/BM25"
                  >
                    <Trash2 size={14} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
