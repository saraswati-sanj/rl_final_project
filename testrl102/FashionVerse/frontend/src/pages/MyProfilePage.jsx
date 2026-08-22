import React, { useState, useEffect } from 'react';
import { User, Sparkles, Sliders, ThumbsUp, Heart, ShoppingBag, Bookmark, Zap } from 'lucide-react';
import { getUserProfile } from '../services/api';

export default function MyProfilePage({ userId = 'user_default' }) {
  const [profile, setProfile] = useState(null);

  useEffect(() => {
    getUserProfile(userId).then(setProfile).catch(console.error);
  }, [userId]);

  if (!profile) {
    return (
      <div style={{ maxWidth: '1000px', margin: '3rem auto', textAlign: 'center', color: '#9aa0b8' }}>
        Loading user preference model...
      </div>
    );
  }

  const styles = profile.style_estimates || {};
  const colors = profile.color_estimates || {};

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '1.5rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
          <h2 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Adaptive User Profile</h2>
          <span className="badge badge-rl">RL Agent Belief State</span>
        </div>
        <p style={{ color: '#9aa0b8', fontSize: '0.95rem' }}>
          The observable preference state dynamically estimated and refined by the RL agent via Exponential Moving Average (EMA) updates.
        </p>
      </div>

      {/* Top Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#9aa0b8', fontSize: '0.8rem', fontWeight: 600 }}>
            <Zap size={16} color="#6C63FF" /> TOTAL INTERACTIONS
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, marginTop: '0.3rem' }}>
            {profile.total_interactions}
          </div>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#9aa0b8', fontSize: '0.8rem', fontWeight: 600 }}>
            <ThumbsUp size={16} color="#55efc4" /> ACCEPTANCE RATE
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#55efc4', marginTop: '0.3rem' }}>
            {Math.round(profile.acceptance_rate * 100)}%
          </div>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#9aa0b8', fontSize: '0.8rem', fontWeight: 600 }}>
            <Heart size={16} color="#ff6584" /> LIKES / LOVES
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#ff6584', marginTop: '0.3rem' }}>
            {profile.likes}
          </div>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#9aa0b8', fontSize: '0.8rem', fontWeight: 600 }}>
            <ShoppingBag size={16} color="#fed330" /> PURCHASES / SAVES
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#fed330', marginTop: '0.3rem' }}>
            {profile.purchases + profile.saves}
          </div>
        </div>
      </div>

      {/* Style & Color Preferences Breakdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Style Preferences */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sliders size={18} color="#6C63FF" />
            Estimated Style Affinities (0.0 – 1.0)
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
            {Object.entries(styles).map(([styleKey, val]) => (
              <div key={styleKey}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                  <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{styleKey.replace('_', ' ')}</span>
                  <span style={{ color: '#6C63FF', fontWeight: 700 }}>{(val * 100).toFixed(0)}%</span>
                </div>
                <div style={{ width: '100%', height: '7px', background: 'rgba(255,255,255,0.06)', borderRadius: '9999px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(val * 100, 100)}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #6C63FF 0%, #a29bfe 100%)',
                    borderRadius: '9999px',
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Color Preferences */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={18} color="#FF6584" />
            Estimated Color Affinities (0.0 – 1.0)
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
            {Object.entries(colors).map(([colorKey, val]) => (
              <div key={colorKey}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                  <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{colorKey.replace('_', ' ')}</span>
                  <span style={{ color: '#FF6584', fontWeight: 700 }}>{(val * 100).toFixed(0)}%</span>
                </div>
                <div style={{ width: '100%', height: '7px', background: 'rgba(255,255,255,0.06)', borderRadius: '9999px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(val * 100, 100)}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #FF6584 0%, #FF8E53 100%)',
                    borderRadius: '9999px',
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
