import React from 'react';
import { FileText, Layers, Image, Database } from 'lucide-react';

export default function StatsCards({ stats, loading }) {
  const cards = [
    {
      title: 'Total Documents',
      value: stats?.total_documents ?? 0,
      icon: <FileText size={22} className="text-green-accent" />,
      desc: 'Datasheets & web scrapings',
    },
    {
      title: 'Semantic Chunks',
      value: stats?.total_chunks ?? 0,
      icon: <Layers size={22} className="text-green-accent" />,
      desc: 'Token-limited retrieval units',
    },
    {
      title: 'Extracted Diagrams',
      value: stats?.total_images ?? 0,
      icon: <Image size={22} className="text-green-accent" />,
      desc: 'CLIP embedded visual models',
    },
    {
      title: 'Ready Sources',
      value: stats?.ready_count ?? 0,
      icon: <Database size={22} className="text-green-accent" />,
      desc: 'Active retrieval sources',
    },
  ];

  return (
    <div className="admin-stats-grid">
      {cards.map((card, i) => (
        <div key={i} className="admin-stat-card">
          <div className="stat-card-header">
            <span className="stat-card-title">{card.title}</span>
            <div className="stat-card-icon">{card.icon}</div>
          </div>
          <div className="stat-card-body">
            {loading ? (
              <div className="stat-loading-pulse">...</div>
            ) : (
              <span className="stat-card-value">
                {card.value.toLocaleString()}
              </span>
            )}
            <span className="stat-card-desc">{card.desc}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
