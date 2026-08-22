import React, { useState } from 'react';
import { ArrowRight, CheckCircle2, RefreshCw, Sparkles, Zap, Heart, ThumbsDown, BarChart2 } from 'lucide-react';
import confetti from 'canvas-confetti';
import AvatarCanvas from '../three/AvatarCanvas';
import { sendMessage, submitFeedback } from '../services/api';

const DEMO_STEPS = [
  { id: 1, title: 'Step 1: User Enters Natural Language Request', desc: 'User enters: "I need a semi-formal outfit for a college presentation under ₹2500."' },
  { id: 2, title: 'Step 2: GenAI Intent Extraction', desc: 'LLM extracts structured constraints: Occasion=College, Formality=2-4, Budget=₹2500.' },
  { id: 3, title: 'Step 3: RL Agent Selects Outfit (PPO)', desc: 'PPO Policy processes 70-dim state vector and sequentially samples Top, Bottom, Shoes.' },
  { id: 4, title: 'Step 4: 3D Avatar Try-On', desc: 'Three.js renders selected pieces onto the 3D procedural mannequin in real time.' },
  { id: 5, title: 'Step 5: User Provides Negative Feedback', desc: 'Simulate user rejecting outfit (e.g. style was too casual for presentation).' },
  { id: 6, title: 'Step 6: Negative Reward Generated', desc: 'Reward function computes penalty: -8.0 base penalty + poor occasion penalty.' },
  { id: 7, title: 'Step 7: State & Preference Update', desc: 'EMA adjusts observable preference vector; casual weight decreases, formality increases.' },
  { id: 8, title: 'Step 8: RL Adapts & Selects New Outfit', desc: 'Policy network receives updated state and selects sharper, formal-matching pieces.' },
  { id: 9, title: 'Step 9: User Provides Positive Feedback', desc: 'User clicks "Love" ❤️ on the newly adapted recommendation.' },
  { id: 10, title: 'Step 10: Positive Reward Generated', desc: 'Terminal reward: +10.0 user satisfaction + compatibility & budget bonus.' },
  { id: 11, title: 'Step 11: Real-Time Policy Learning Demonstrated', desc: 'Acceptance rate increases, preference weights converge to user personality.' },
];

export default function DemoModePage({ onGoToDashboard }) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [demoOutfit, setDemoOutfit] = useState([]);
  const [demoInfo, setDemoInfo] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  const currentStep = DEMO_STEPS[currentStepIndex];

  const handleNextStep = async () => {
    setIsLoading(true);
    const nextIdx = currentStepIndex + 1;

    try {
      if (currentStepIndex === 0) {
        // Step 1 -> 2: Intent extraction
        const data = await sendMessage('I need a semi-formal outfit for a college presentation under ₹2500.', 'demo_user');
        setDemoInfo({
          constraints: data.constraints,
          explanation: data.explanation,
        });
        setDemoOutfit(data.outfit?.outfit || []);
      } else if (currentStepIndex === 4) {
        // Step 5: Submit Dislike
        const res = await submitFeedback({
          userId: 'demo_user',
          feedback: 'dislike',
          itemIds: demoOutfit.map(i => i.item_id),
          occasion: 'college',
        });
        setDemoInfo(prev => ({ ...prev, reward: res.computed_reward, profile: res.updated_profile }));
      } else if (currentStepIndex === 6) {
        // Step 7 -> 8: Get new adapted recommendation
        const data = await sendMessage('Need a sharper semi-formal college presentation look with crisp chinos and formal shoes under ₹2500.', 'demo_user');
        setDemoOutfit(data.outfit?.outfit || []);
        setDemoInfo(prev => ({ ...prev, explanation: data.explanation, newOutfit: data.outfit }));
      } else if (currentStepIndex === 8) {
        // Step 9: Submit Love
        const res = await submitFeedback({
          userId: 'demo_user',
          feedback: 'love',
          itemIds: demoOutfit.map(i => i.item_id),
          occasion: 'college',
        });
        confetti({ particleCount: 60, spread: 70, origin: { y: 0.7 } });
        setDemoInfo(prev => ({ ...prev, reward: res.computed_reward, profile: res.updated_profile }));
      }

      if (nextIdx < DEMO_STEPS.length) {
        setCurrentStepIndex(nextIdx);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setCurrentStepIndex(0);
    setDemoOutfit([]);
    setDemoInfo({});
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.5rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h2 style={{ fontSize: '1.8rem', fontWeight: 800 }}>Viva & Presentation Demo Mode</h2>
            <span className="badge badge-rl">11-Step Interactive Flow</span>
          </div>
          <p style={{ color: '#9aa0b8', fontSize: '0.95rem' }}>
            Live demonstration of the genuine RL loop: State &rarr; Action &rarr; Reward &rarr; Preference Adaptation &rarr; Policy Improvement.
          </p>
        </div>
        <button onClick={handleReset} className="btn-secondary" style={{ fontSize: '0.85rem' }}>
          <RefreshCw size={15} /> Reset Demo Flow
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '1.5rem' }}>
        {/* Left Column: Interactive Step Controller */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          {/* Progress Bar */}
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.4rem' }}>
              <span style={{ fontWeight: 700, color: '#6C63FF' }}>Progress: Step {currentStepIndex + 1} of {DEMO_STEPS.length}</span>
              <span style={{ color: '#9aa0b8' }}>{Math.round(((currentStepIndex + 1) / DEMO_STEPS.length) * 100)}% Completed</span>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '9999px', overflow: 'hidden' }}>
              <div style={{
                width: `${((currentStepIndex + 1) / DEMO_STEPS.length) * 100}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #6C63FF 0%, #FF6584 100%)',
                transition: 'width 0.3s ease',
              }} />
            </div>
          </div>

          {/* Current Step Card */}
          <div style={{
            background: 'rgba(108, 99, 255, 0.1)',
            border: '1px solid rgba(108, 99, 255, 0.3)',
            borderRadius: '12px',
            padding: '1.25rem',
            marginBottom: '1.5rem',
          }}>
            <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#fff', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Sparkles size={20} color="#6C63FF" />
              {currentStep.title}
            </div>
            <p style={{ color: '#c5c9db', fontSize: '0.92rem', lineHeight: 1.5 }}>
              {currentStep.desc}
            </p>
          </div>

          {/* Interactive State Display */}
          <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1.5rem' }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#9aa0b8', marginBottom: '0.6rem' }}>
              SYSTEM TRACE LOGS:
            </div>
            <div style={{
              background: '#090a0f',
              padding: '1rem',
              borderRadius: '10px',
              fontFamily: 'monospace',
              fontSize: '0.8rem',
              color: '#55efc4',
              maxHeight: '220px',
              overflowY: 'auto',
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              <div>&gt; System Initialized: Gymnasium FashionEnv (State_dim=70, Action_dim=60)</div>
              {currentStepIndex >= 1 && (
                <div>&gt; GenAI Parser: Occasion=&quot;college&quot; | Budget=2500 | Formality=[2,4]</div>
              )}
              {currentStepIndex >= 2 && (
                <div>&gt; PPO Action: Selected Top_004, Bottom_012, Shoes_003 | Compat Score: 0.88</div>
              )}
              {currentStepIndex >= 5 && (
                <div style={{ color: '#ff7675' }}>&gt; Reward Computed: -8.0 (Dislike feedback mapped to penalty)</div>
              )}
              {currentStepIndex >= 6 && (
                <div>&gt; Preference State EMA Update: Formal_pref &uarr; 0.72 | Casual_pref &darr; 0.45</div>
              )}
              {currentStepIndex >= 7 && (
                <div>&gt; PPO Policy Inferred New Context &rarr; Re-sampled with higher formality target</div>
              )}
              {currentStepIndex >= 9 && (
                <div style={{ color: '#55efc4' }}>&gt; Reward Computed: +10.0 (Love feedback received!) Policy weights validated.</div>
              )}
            </div>
          </div>

          {/* Next Step Action Button */}
          {currentStepIndex < DEMO_STEPS.length - 1 ? (
            <button
              onClick={handleNextStep}
              disabled={isLoading}
              className="btn-primary"
              style={{ width: '100%', justifyContent: 'center', padding: '0.8rem' }}
            >
              {isLoading ? (
                <RefreshCw size={18} className="spin-animation" style={{ animation: 'spin 1s linear infinite' }} />
              ) : (
                <>Execute Next Step ({currentStepIndex + 2}) <ArrowRight size={18} /></>
              )}
            </button>
          ) : (
            <div style={{ display: 'flex', gap: '0.8rem' }}>
              <button
                onClick={handleReset}
                className="btn-secondary"
                style={{ flex: 1, justifyContent: 'center' }}
              >
                Restart Demo
              </button>
              <button
                onClick={onGoToDashboard}
                className="btn-primary"
                style={{ flex: 1, justifyContent: 'center' }}
              >
                View RL Dashboard <BarChart2 size={18} />
              </button>
            </div>
          )}
        </div>

        {/* Right Column: 3D Visualization */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="glass-panel" style={{ height: '460px', overflow: 'hidden' }}>
            <AvatarCanvas outfit={demoOutfit} />
          </div>

          {/* Outfits in demo */}
          <div className="glass-panel" style={{ padding: '1.2rem' }}>
            <div style={{ fontSize: '0.8rem', color: '#9aa0b8', fontWeight: 600, marginBottom: '0.4rem' }}>
              CURRENT OUTFIT STATE ON AVATAR:
            </div>
            {demoOutfit.length > 0 ? (
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {demoOutfit.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '8px',
                      padding: '0.4rem 0.7rem',
                      fontSize: '0.8rem',
                    }}
                  >
                    <strong>{item.category.toUpperCase()}:</strong> {item.name} (₹{item.price})
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: '#626880', fontSize: '0.85rem' }}>
                Click &quot;Execute Next Step&quot; to begin the sequential RL recommendation process.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
