import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, RefreshCw, Layers } from 'lucide-react';
import { getAdminDocuments, getAdminStatistics } from '../services/api';
import StatsCards from '../components/admin/StatsCards';
import UploadPanel from '../components/admin/UploadPanel';
import DocumentTable from '../components/admin/DocumentTable';

export default function AdminPage({ onBackToChat }) {
  // Authorization States
  const [isAuthorized, setIsAuthorized] = useState(false);
  const [secretInput, setSecretInput] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [validating, setValidating] = useState(false);

  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const pollTimerRef = useRef(null);

  const validateSecret = async (secret) => {
    setValidating(true);
    setErrorMsg('');
    localStorage.setItem('admin_secret_key', secret);
    try {
      const [docsData, statsData] = await Promise.all([
        getAdminDocuments(),
        getAdminStatistics(),
      ]);
      setDocuments(docsData);
      setStats(statsData);
      setIsAuthorized(true);
    } catch (err) {
      console.error('Validation error:', err);
      setErrorMsg('Invalid Admin Secret Key. Access Denied.');
      localStorage.removeItem('admin_secret_key');
      setIsAuthorized(false);
    } finally {
      setValidating(false);
      setLoading(false);
    }
  };

  const fetchRegistryData = async (showRefreshIndicator = false) => {
    if (showRefreshIndicator) setRefreshing(true);
    try {
      const [docsData, statsData] = await Promise.all([
        getAdminDocuments(),
        getAdminStatistics(),
      ]);
      setDocuments(docsData);
      setStats(statsData);
    } catch (e) {
      console.error('Failed to load admin registry logs:', e);
      if (e.response && (e.response.status === 401 || e.response.status === 403)) {
        setIsAuthorized(false);
        localStorage.removeItem('admin_secret_key');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Initial Data Fetch
  useEffect(() => {
    const storedSecret = localStorage.getItem('admin_secret_key');
    if (storedSecret) {
      validateSecret(storedSecret);
    } else {
      setLoading(false);
    }
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  // Intelligent Polling Mechanism
  useEffect(() => {
    if (!isAuthorized) return;
    
    const hasActiveBackgroundTasks = documents.some(
      (doc) => doc.status === 'processing' || doc.status === 'deleting'
    );

    if (hasActiveBackgroundTasks) {
      if (!pollTimerRef.current) {
        pollTimerRef.current = setInterval(() => {
          fetchRegistryData();
        }, 2500);
      }
    } else {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    }

    return () => {
      if (pollTimerRef.current && !hasActiveBackgroundTasks) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [documents, isAuthorized]);

  const handleUpdateTrigger = () => {
    fetchRegistryData(true);
  };

  const handleUnlock = (e) => {
    e.preventDefault();
    if (!secretInput.trim()) return;
    validateSecret(secretInput.trim());
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_secret_key');
    setIsAuthorized(false);
    setSecretInput('');
  };

  const handleRealLogout = () => {
    localStorage.removeItem('admin_secret_key');
    window.location.href = '/';
  };

  if (loading && !isAuthorized) {
    return (
      <div className="admin-loading-container">
        <div className="loader-spinner"></div>
        <p>Verifying credentials...</p>
      </div>
    );
  }

  if (!isAuthorized) {
    return (
      <div className="admin-lock-overlay">
        <div className="admin-lock-card">
          <div className="lock-icon">🔒</div>
          <h2>Admin Authentication</h2>
          <p>Please enter the secret key to unlock the admin dashboard.</p>
          
          <form onSubmit={handleUnlock}>
            <input
              type="password"
              placeholder="Enter Admin Secret Key"
              value={secretInput}
              onChange={(e) => setSecretInput(e.target.value)}
              className="admin-lock-input"
              disabled={validating}
              autoFocus
            />
            {errorMsg && <div className="error-message">{errorMsg}</div>}
            <div className="lock-actions">
              <button type="button" className="btn-secondary" onClick={onBackToChat}>
                Back to Chat
              </button>
              <button type="submit" className="btn-primary" disabled={validating}>
                {validating ? 'Unlocking...' : 'Unlock'}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="admin-page-container">
      {/* Top Header Navigation */}
      <header className="admin-header">
        <div className="admin-header-left">
          <button onClick={onBackToChat} className="back-chat-link" title="Return to Chat">
            <ArrowLeft size={16} />
            <span>Chat Assistant</span>
          </button>
          <div className="admin-brand">
            <Layers size={18} className="text-green-accent" />
            <h1>Knowledge Management Admin</h1>
          </div>
        </div>
        <div className="admin-header-right" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={handleLogout}
            className="back-chat-link"
            title="Lock Admin Dashboard"
          >
            Lock Panel
          </button>
          <button
            onClick={handleRealLogout}
            className="back-chat-link"
            title="Logout and return to Home"
          >
            Logout
          </button>
          <button
            onClick={() => handleUpdateTrigger()}
            className={`admin-icon-btn ${refreshing ? 'rotating' : ''}`}
            disabled={refreshing}
            title="Force refresh database status"
          >
            <RefreshCw size={15} />
          </button>
        </div>
      </header>

      {/* Main Grid Body */}
      <main className="admin-main">
        {/* Left Stats Section */}
        <section className="admin-stats-section">
          <h2 className="section-title-tag">Overview Statistics</h2>
          <StatsCards stats={stats} loading={loading} />
        </section>

        {/* Dynamic Dual columns layout */}
        <div className="admin-grid-layout">
          {/* Form Actions Section */}
          <section className="admin-actions-section">
            <h2 className="section-title-tag">Ingest Knowledge Source</h2>
            <UploadPanel onIngestionTriggered={handleUpdateTrigger} />
          </section>

          {/* Registry Logs Section */}
          <section className="admin-registry-section">
            <h2 className="section-title-tag">Indexed Source Registry</h2>
            <DocumentTable
              documents={documents}
              loading={loading}
              onUpdate={handleUpdateTrigger}
            />
          </section>
        </div>
      </main>
    </div>
  );
}
