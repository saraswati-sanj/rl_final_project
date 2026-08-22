import React, { useState, useEffect } from 'react';
import { Shirt, Glasses, Sparkles, Check, RefreshCw, ShoppingCart, Heart, ThumbsDown } from 'lucide-react';
import confetti from 'canvas-confetti';
import AvatarCanvas from '../three/AvatarCanvas';
import { getRecommendation, submitFeedback } from '../services/api';

const CATEGORIES = ['All', 'top', 'bottom', 'dress', 'shoes', 'accessory'];

export default function TryOnPage({ userId = 'user_default' }) {
  const [outfit, setOutfit] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState('');

  const fetchNewOutfit = async (occasion = 'casual') => {
    setIsLoading(true);
    try {
      const data = await getRecommendation({ user_id: userId, occasion, budget: 3000 });
      setOutfit(data.outfit || []);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchNewOutfit('casual');
  }, []);

  const handleFeedback = async (type) => {
    try {
      const res = await submitFeedback({
        userId,
        outfitId: `tryon_${Date.now()}`,
        feedback: type,
        itemIds: outfit.map(i => i.item_id),
      });
      setFeedbackStatus(`Recorded ${type.toUpperCase()} feedback (+${res.computed_reward} reward)`);
      if (['love', 'like', 'save', 'purchase'].includes(type)) {
        confetti({ particleCount: 30, spread: 50 });
      }
      setTimeout(() => setFeedbackStatus(''), 4000);
    } catch (e) {
      console.error(e);
    }
  };

  const filteredItems = selectedCategory === 'All'
    ? outfit
    : outfit.filter(i => i.category === selectedCategory);

  const totalPrice = outfit.reduce((acc, i) => acc + i.price, 0);

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.5rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h2 style={{ fontSize: '1.8rem', fontWeight: 800 }}>3D Virtual Try-On Studio</h2>
            <span className="badge badge-success">WebXR Enabled</span>
          </div>
          <p style={{ color: '#9aa0b8', fontSize: '0.95rem' }}>
            Inspect your RL-curated outfit in full 3D, customize items, and experience immersive WebXR virtual reality.
          </p>
        </div>

        <button
          onClick={() => fetchNewOutfit('semi_formal')}
          disabled={isLoading}
          className="btn-primary"
          style={{ fontSize: '0.85rem' }}
        >
          <RefreshCw size={15} className={isLoading ? 'spin-animation' : ''} /> Generate New Look (PPO)
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem' }}>
        {/* Left: Big 3D Stage */}
        <div className="glass-panel" style={{ height: '620px', position: 'relative', overflow: 'hidden' }}>
          <AvatarCanvas outfit={outfit} />
        </div>

        {/* Right: Outfit Breakdown & Wardrobe Swapper */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          {/* Summary Box */}
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#9aa0b8', fontWeight: 600 }}>CURRENT TOTAL PRICE</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#55efc4' }}>₹{totalPrice}</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: '#9aa0b8', fontWeight: 600 }}>COMPATIBILITY RATING</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#a29bfe' }}>94%</div>
              </div>
            </div>

            {/* Feedback Actions */}
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
              <button
                onClick={() => handleFeedback('love')}
                className="btn-primary"
                style={{ flex: 1, justifyContent: 'center', background: 'linear-gradient(135deg, #FF6584 0%, #FF8E53 100%)' }}
              >
                <Heart size={16} /> Love It
              </button>
              <button
                onClick={() => handleFeedback('purchase')}
                className="btn-primary"
                style={{ flex: 1, justifyContent: 'center' }}
              >
                <ShoppingCart size={16} /> Buy Look
              </button>
              <button
                onClick={() => handleFeedback('dislike')}
                className="btn-secondary"
                style={{ padding: '0.65rem 1rem' }}
                title="Dislike / Replace"
              >
                <ThumbsDown size={16} />
              </button>
            </div>

            {feedbackStatus && (
              <div style={{ marginTop: '0.6rem', color: '#55efc4', fontSize: '0.8rem', textAlign: 'center', fontWeight: 600 }}>
                {feedbackStatus}
              </div>
            )}
          </div>

          {/* Category Filter */}
          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
            {CATEGORIES.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                style={{
                  background: selectedCategory === cat ? '#6C63FF' : 'rgba(255, 255, 255, 0.05)',
                  color: selectedCategory === cat ? '#fff' : '#9aa0b8',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '9999px',
                  padding: '0.35rem 0.85rem',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  fontWeight: 600,
                  textTransform: 'capitalize',
                }}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Item Cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', overflowY: 'auto', maxHeight: '350px' }}>
            {filteredItems.map((item, idx) => (
              <div
                key={idx}
                className="glass-card"
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.9rem 1.1rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{
                    width: '14px',
                    height: '14px',
                    borderRadius: '50%',
                    background: item.color === 'black' ? '#222' : (item.color === 'white' ? '#eee' : item.color),
                    border: '1px solid rgba(255,255,255,0.4)',
                  }} />
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{item.name}</div>
                    <div style={{ fontSize: '0.75rem', color: '#9aa0b8' }}>
                      {item.category.toUpperCase()} &bull; Style: {item.style} &bull; Color: {item.color}
                    </div>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontWeight: 800, color: '#55efc4', fontSize: '1rem' }}>₹{item.price}</div>
                  <div style={{ fontSize: '0.7rem', color: '#6C63FF' }}>RL Selected</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
