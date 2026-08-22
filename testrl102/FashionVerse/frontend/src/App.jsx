import React, { useState } from 'react';
import Navbar from './components/Navbar';
import HomeStylistPage from './pages/HomeStylistPage';
import TryOnPage from './pages/TryOnPage';
import DemoModePage from './pages/DemoModePage';
import RLDashboardPage from './pages/RLDashboardPage';
import MyProfilePage from './pages/MyProfilePage';

export default function App() {
  const [activeTab, setActiveTab] = useState('stylist');
  const [userId, setUserId] = useState('user_fashion_01');

  return (
    <div className="app-container">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {activeTab === 'stylist' && <HomeStylistPage userId={userId} onGoToTryOn={() => setActiveTab('tryon')} />}
        {activeTab === 'tryon' && <TryOnPage userId={userId} />}
        {activeTab === 'demo' && <DemoModePage onGoToDashboard={() => setActiveTab('dashboard')} />}
        {activeTab === 'dashboard' && <RLDashboardPage />}
        {activeTab === 'profile' && <MyProfilePage userId={userId} />}
      </main>

      {/* Footer */}
      <footer style={{
        textAlign: 'center',
        padding: '1.5rem',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        color: '#626880',
        fontSize: '0.8rem',
      }}>
        FashionVerse &copy; 2026 &bull; Adaptive AI Fashion Stylist using PPO Reinforcement Learning & GenAI &bull; Academically Defensible Local-First Implementation
      </footer>
    </div>
  );
}
