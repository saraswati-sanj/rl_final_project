import React, { useState, useEffect } from 'react';
import { Send, Heart, ThumbsUp, Meh, ThumbsDown, Bookmark, ShoppingBag, Sparkles, RefreshCw, Zap } from 'lucide-react';
import confetti from 'canvas-confetti';
import AvatarCanvas from '../three/AvatarCanvas';
import { sendMessage, submitFeedback } from '../services/api';

const QUICK_PROMPTS = [
  "I need a semi-formal outfit for a college presentation under ₹2500.",
  "Looking for a casual summer streetwear look under ₹2000.",
  "Formal office meeting look with neutral colors under ₹4000.",
  "Party outfit for a weekend night out under ₹3000.",
];

export default function HomeStylistPage({ userId = 'user_default', onGoToTryOn }) {
  const [messages, setMessages] = useState([
    {
      sender: 'ai',
      text: 'Hello! I am your FashionVerse AI Stylist powered by Reinforcement Learning (PPO) and GenAI. Describe your occasion, style, and budget, and our RL agent will construct your optimal outfit in real time!',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentOutfit, setCurrentOutfit] = useState([]);
  const [currentExplanation, setCurrentExplanation] = useState('');
  const [currentConstraints, setCurrentConstraints] = useState(null);
  const [compatibilityScore, setCompatibilityScore] = useState(0);
  const [totalPrice, setTotalPrice] = useState(0);
  const [lastRewardInfo, setLastRewardInfo] = useState(null);

  // Load an initial default outfit recommendation on mount
  useEffect(() => {
    handleSendPrompt(QUICK_PROMPTS[0], false);
  }, []);

  const handleSendPrompt = async (promptText, addAsUserMsg = true) => {
    if (!promptText.trim()) return;

    if (addAsUserMsg) {
      setMessages((prev) => [
        ...prev,
        {
          sender: 'user',
          text: promptText,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    }
    setInputText('');
    setIsLoading(true);

    try {
      const data = await sendMessage(promptText, userId);
      const outfitList = data.outfit?.outfit || [];

      setCurrentOutfit(outfitList);
      setCurrentExplanation(data.explanation || data.reply);
      setCurrentConstraints(data.constraints);
      setCompatibilityScore(data.outfit?.compatibility_score || 0.85);
      setTotalPrice(data.outfit?.total_price || outfitList.reduce((acc, i) => acc + i.price, 0));

      setMessages((prev) => [
        ...prev,
        {
          sender: 'ai',
          text: data.reply,
          explanation: data.explanation,
          constraints: data.constraints,
          outfit: outfitList,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: 'ai',
          text: `Error connecting to backend: ${err.message}. Please ensure the FastAPI server is running on port 8000.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (feedbackType) => {
    if (!currentOutfit.length) return;

    const itemIds = currentOutfit.map((i) => i.item_id);
    try {
      const res = await submitFeedback({
        userId,
        outfitId: `outfit_${Date.now()}`,
        feedback: feedbackType,
        itemIds,
        occasion: currentConstraints?.occasion || 'casual',
      });

      setLastRewardInfo({
        feedback: feedbackType,
        reward: res.computed_reward,
        acceptanceRate: res.acceptance_rate,
      });

      if (['love', 'like', 'save', 'purchase'].includes(feedbackType)) {
        confetti({ particleCount: 40, spread: 60, origin: { y: 0.8 } });
      }

      setTimeout(() => setLastRewardInfo(null), 5000);
    } catch (err) {
      console.error('Feedback submission failed:', err);
    }
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.5rem 2rem' }}>
      {/* Page Header */}
      <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Adaptive AI Fashion Stylist</h2>
          <p style={{ color: '#9aa0b8', fontSize: '0.95rem' }}>
            Natural Language (GenAI) &rarr; Sequential Decision Making (PPO RL) &rarr; 3D Visual Try-On
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-genai">GenAI Intent Parser</span>
          <span className="badge badge-rl">PPO Policy Network</span>
          <span className="badge badge-success">3D/WebXR Engine</span>
        </div>
      </div>

      {/* Main Grid: Chat Left + 3D Viewport Right */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1.1fr 1fr',
        gap: '1.5rem',
        minHeight: '620px',
      }}>
        {/* Left Column: Conversational Stylist */}
        <div className="glass-panel" style={{
          display: 'flex',
          flexDirection: 'column',
          height: '660px',
          padding: '1.25rem',
        }}>
          {/* Quick Prompts */}
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: '0.8rem', color: '#9aa0b8', marginBottom: '0.5rem', fontWeight: 600 }}>
              💡 Quick Prompts:
            </div>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              {QUICK_PROMPTS.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSendPrompt(p)}
                  disabled={isLoading}
                  style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    borderRadius: '9999px',
                    color: '#c5c9db',
                    fontSize: '0.75rem',
                    padding: '0.35rem 0.75rem',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = 'rgba(108, 99, 255, 0.2)';
                    e.target.style.borderColor = 'rgba(108, 99, 255, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'rgba(255, 255, 255, 0.05)';
                    e.target.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                  }}
                >
                  {p.slice(0, 45)}...
                </button>
              ))}
            </div>
          </div>

          {/* Messages Area */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            paddingRight: '0.5rem',
          }}>
            {messages.map((m, idx) => (
              <div
                key={idx}
                style={{
                  alignSelf: m.sender === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '85%',
                  background: m.sender === 'user' ? 'linear-gradient(135deg, #6C63FF 0%, #7D75FF 100%)' : 'rgba(18, 20, 31, 0.85)',
                  border: m.sender === 'user' ? 'none' : '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '14px',
                  padding: '0.9rem 1.1rem',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                }}
              >
                <div style={{ fontSize: '0.92rem', lineHeight: 1.5 }}>
                  {m.text}
                </div>

                {/* Structured Intent Breakdown badge */}
                {m.constraints && (
                  <div style={{
                    marginTop: '0.6rem',
                    padding: '0.5rem 0.75rem',
                    background: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '8px',
                    fontSize: '0.75rem',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '0.5rem',
                  }}>
                    <span>🎯 Occasion: <strong style={{ color: '#48dbfb' }}>{m.constraints.occasion}</strong></span>
                    <span>💰 Budget: <strong style={{ color: '#55efc4' }}>₹{m.constraints.budget}</strong></span>
                    <span>👔 Formality: <strong style={{ color: '#fed330' }}>{m.constraints.formality_min}-{m.constraints.formality_max}/5</strong></span>
                  </div>
                )}

                <div style={{ fontSize: '0.7rem', color: m.sender === 'user' ? 'rgba(255,255,255,0.7)' : '#626880', marginTop: '0.4rem', textAlign: 'right' }}>
                  {m.timestamp}
                </div>
              </div>
            ))}
            {isLoading && (
              <div style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#6C63FF' }}>
                <RefreshCw size={16} className="spin-animation" style={{ animation: 'spin 1s linear infinite' }} />
                <span style={{ fontSize: '0.85rem' }}>RL Agent evaluating candidate action spaces...</span>
              </div>
            )}
          </div>

          {/* Input Box */}
          <div style={{
            marginTop: '1rem',
            display: 'flex',
            gap: '0.5rem',
            background: 'rgba(10, 11, 16, 0.6)',
            padding: '0.4rem',
            borderRadius: '9999px',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendPrompt(inputText)}
              placeholder="e.g. Semi-formal dinner outfit under ₹3000..."
              disabled={isLoading}
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                color: '#fff',
                padding: '0.6rem 1.2rem',
                fontSize: '0.95rem',
                outline: 'none',
              }}
            />
            <button
              onClick={() => handleSendPrompt(inputText)}
              disabled={isLoading || !inputText.trim()}
              className="btn-primary"
              style={{ padding: '0.6rem 1.2rem' }}
            >
              <Send size={16} />
            </button>
          </div>
        </div>

        {/* Right Column: 3D Try-On Viewport & Feedback Bar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* 3D Canvas Box */}
          <div className="glass-panel" style={{ height: '420px', overflow: 'hidden' }}>
            <AvatarCanvas outfit={currentOutfit} />
          </div>

          {/* Decision Metrics & Items Card */}
          <div className="glass-panel" style={{ padding: '1.2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#9aa0b8', fontWeight: 600 }}>RL RECOMMENDED OUTFIT</span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                  Total: <span style={{ color: '#55efc4' }}>₹{totalPrice}</span>
                  {currentConstraints?.budget && (
                    <span style={{ fontSize: '0.8rem', color: '#9aa0b8', marginLeft: '0.5rem' }}>
                      (Saved ₹{currentConstraints.budget - totalPrice})
                    </span>
                  )}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: '#9aa0b8', fontWeight: 600 }}>COMPATIBILITY</span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#a29bfe' }}>
                  {Math.round(compatibilityScore * 100)}%
                </div>
              </div>
            </div>

            {/* Item Pills */}
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
              {currentOutfit.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    borderRadius: '8px',
                    padding: '0.4rem 0.7rem',
                    fontSize: '0.8rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                  }}
                >
                  <span style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: item.color === 'black' ? '#222' : (item.color === 'white' ? '#eee' : item.color),
                    border: '1px solid rgba(255,255,255,0.3)',
                    display: 'inline-block',
                  }} />
                  <strong>{item.category.toUpperCase()}:</strong> {item.name} (₹{item.price})
                </div>
              ))}
            </div>

            {/* Live RL Feedback Actions */}
            <div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '0.5rem',
              }}>
                <span style={{ fontSize: '0.75rem', color: '#9aa0b8', fontWeight: 600 }}>
                  TRAIN THE RL AGENT (User Feedback &rarr; Reward &rarr; Policy Update):
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '0.4rem' }}>
                <button
                  onClick={() => handleFeedback('love')}
                  title="Love (+10 reward)"
                  style={{
                    background: 'rgba(255, 101, 132, 0.15)',
                    color: '#ff6584',
                    border: '1px solid rgba(255, 101, 132, 0.3)',
                    borderRadius: '8px',
                    padding: '0.5rem 0',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                  }}
                >
                  <Heart size={16} /> Love
                </button>
                <button
                  onClick={() => handleFeedback('like')}
                  title="Like (+5 reward)"
                  style={{
                    background: 'rgba(67, 184, 156, 0.15)',
                    color: '#43b89c',
                    border: '1px solid rgba(67, 184, 156, 0.3)',
                    borderRadius: '8px',
                    padding: '0.5rem 0',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                  }}
                >
                  <ThumbsUp size={16} /> Like
                </button>
                <button
                  onClick={() => handleFeedback('neutral')}
                  title="Neutral (0 reward)"
                  style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    color: '#9aa0b8',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    padding: '0.5rem 0',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                  }}
                >
                  <Meh size={16} /> Neutral
                </button>
                <button
                  onClick={() => handleFeedback('dislike')}
                  title="Dislike (-8 penalty reward)"
                  style={{
                    background: 'rgba(235, 77, 75, 0.15)',
                    color: '#eb4d4b',
                    border: '1px solid rgba(235, 77, 75, 0.3)',
                    borderRadius: '8px',
                    padding: '0.5rem 0',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                  }}
                >
                  <ThumbsDown size={16} /> Dislike
                </button>
                <button
                  onClick={() => handleFeedback('save')}
                  title="Save (+7 reward)"
                  style={{
                    background: 'rgba(245, 166, 35, 0.15)',
                    color: '#f5a623',
                    border: '1px solid rgba(245, 166, 35, 0.3)',
                    borderRadius: '8px',
                    padding: '0.5rem 0',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                  }}
                >
                  <Bookmark size={16} /> Save
                </button>
                <button
                  onClick={() => handleFeedback('purchase')}
                  title="Purchase (+15 reward)"
                  style={{
                    background: 'linear-gradient(135deg, rgba(108,99,255,0.2) 0%, rgba(255,101,132,0.2) 100%)',
                    color: '#fff',
                    border: '1px solid rgba(108,99,255,0.4)',
                    borderRadius: '8px',
                    padding: '0.5rem 0',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                  }}
                >
                  <ShoppingBag size={16} /> Buy
                </button>
              </div>

              {/* Reward Notification Banner */}
              {lastRewardInfo && (
                <div style={{
                  marginTop: '0.8rem',
                  background: lastRewardInfo.reward >= 0 ? 'rgba(67, 184, 156, 0.2)' : 'rgba(235, 77, 75, 0.2)',
                  border: `1px solid ${lastRewardInfo.reward >= 0 ? 'rgba(67, 184, 156, 0.4)' : 'rgba(235, 77, 75, 0.4)'}`,
                  borderRadius: '8px',
                  padding: '0.5rem 0.8rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.8rem',
                  animation: 'fadeIn 0.3s ease',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Zap size={15} color={lastRewardInfo.reward >= 0 ? '#55efc4' : '#ff7675'} />
                    <span>
                      Feedback <strong>{lastRewardInfo.feedback.toUpperCase()}</strong> recorded!
                    </span>
                  </div>
                  <div>
                    RL Reward: <strong style={{ color: lastRewardInfo.reward >= 0 ? '#55efc4' : '#ff7675' }}>
                      {lastRewardInfo.reward > 0 ? `+${lastRewardInfo.reward}` : lastRewardInfo.reward}
                    </strong> | User Acceptance: <strong>{Math.round(lastRewardInfo.acceptanceRate * 100)}%</strong>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
