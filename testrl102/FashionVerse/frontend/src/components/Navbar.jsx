import React from 'react';
import { Sparkles, Shirt, BarChart3, PlayCircle, User, Glasses } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'stylist', label: 'AI Stylist', icon: Sparkles },
    { id: 'tryon', label: '3D Try-On', icon: Shirt },
    { id: 'demo', label: 'Viva Demo Mode', icon: PlayCircle, badge: 'VIVA' },
    { id: 'dashboard', label: 'RL Dashboard', icon: BarChart3 },
    { id: 'profile', label: 'My Style', icon: User },
  ];

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 100,
      background: 'rgba(10, 11, 16, 0.85)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      padding: '0.8rem 2rem',
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        {/* Brand Logo */}
        <div
          onClick={() => setActiveTab('stylist')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            cursor: 'pointer',
          }}
        >
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #6C63FF 0%, #FF6584 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 15px rgba(108, 99, 255, 0.4)',
          }}>
            <Sparkles size={20} color="#fff" />
          </div>
          <div>
            <h1 style={{
              fontSize: '1.25rem',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              background: 'linear-gradient(90deg, #FFFFFF 0%, #D1D5DB 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              FashionVerse
            </h1>
            <div style={{ fontSize: '0.7rem', color: '#6C63FF', fontWeight: 600, letterSpacing: '0.05em' }}>
              RL-PPO + GenAI + 3D/VR
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                style={{
                  background: isActive ? 'rgba(108, 99, 255, 0.15)' : 'transparent',
                  color: isActive ? '#fff' : '#9aa0b8',
                  border: isActive ? '1px solid rgba(108, 99, 255, 0.4)' : '1px solid transparent',
                  borderRadius: '9999px',
                  padding: '0.5rem 1rem',
                  fontSize: '0.88rem',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
              >
                <Icon size={16} color={isActive ? '#6C63FF' : '#9aa0b8'} />
                <span>{item.label}</span>
                {item.badge && (
                  <span style={{
                    fontSize: '0.65rem',
                    background: 'linear-gradient(135deg, #FF6584 0%, #FF8E53 100%)',
                    color: '#fff',
                    padding: '1px 6px',
                    borderRadius: '9999px',
                    fontWeight: 700,
                  }}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Status Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: '#43B89C',
            boxShadow: '0 0 10px #43B89C',
          }} />
          <span style={{ fontSize: '0.8rem', color: '#9aa0b8', fontWeight: 500 }}>
            RL Model: <strong style={{ color: '#55efc4' }}>PPO Online</strong>
          </span>
        </div>
      </div>
    </header>
  );
}
