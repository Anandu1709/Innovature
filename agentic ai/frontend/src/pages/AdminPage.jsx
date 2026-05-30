import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, RefreshCw, Layers } from 'lucide-react';
import { getAdminDocuments, getAdminStatistics } from '../services/api';
import StatsCards from '../components/admin/StatsCards';
import UploadPanel from '../components/admin/UploadPanel';
import DocumentTable from '../components/admin/DocumentTable';

export default function AdminPage({ onBackToChat }) {
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const pollTimerRef = useRef(null);

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
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Initial Data Fetch
  useEffect(() => {
    fetchRegistryData();
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  // Intelligent Polling Mechanism
  // If any document is currently in 'processing' or 'deleting' state,
  // poll every 2.5 seconds to capture background task state transition updates dynamically.
  useEffect(() => {
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
  }, [documents]);

  const handleUpdateTrigger = () => {
    fetchRegistryData(true);
  };

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
        <div className="admin-header-right">
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
