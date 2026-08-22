/**
 * FashionVerse — API Service Client
 * Handles communication with the FastAPI backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function sendMessage(message, userId = 'user_default', budget = 2500, gender = 'unisex') {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, user_id: userId, budget, gender }),
  });
  if (!res.ok) throw new Error(`Chat error: ${res.statusText}`);
  return res.json();
}

export async function getRecommendation(params = {}) {
  const res = await fetch(`${BASE_URL}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Recommend error: ${res.statusText}`);
  return res.json();
}

export async function submitFeedback({
  userId = 'user_default',
  outfitId = 'outfit_current',
  feedback = 'like',
  itemIds = [],
  actionType = 'explicit_feedback',
  occasion = 'casual',
}) {
  const res = await fetch(`${BASE_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      outfit_id: outfitId,
      feedback,
      item_ids: itemIds,
      action_type: actionType,
      occasion,
    }),
  });
  if (!res.ok) throw new Error(`Feedback error: ${res.statusText}`);
  return res.json();
}

export async function getUserProfile(userId = 'user_default') {
  const res = await fetch(`${BASE_URL}/user/${userId}`);
  if (!res.ok) throw new Error(`User profile error: ${res.statusText}`);
  return res.json();
}

export async function getAnalyticsSummary() {
  const res = await fetch(`${BASE_URL}/analytics/summary`);
  if (!res.ok) throw new Error(`Analytics summary error: ${res.statusText}`);
  return res.json();
}

export async function getExperimentResults() {
  const res = await fetch(`${BASE_URL}/analytics/experiments`);
  if (!res.ok) throw new Error(`Experiments error: ${res.statusText}`);
  return res.json();
}

export async function getRecentInteractions() {
  const res = await fetch(`${BASE_URL}/analytics/recent-interactions`);
  if (!res.ok) throw new Error(`Interactions error: ${res.statusText}`);
  return res.json();
}

export async function getRLStatus() {
  const res = await fetch(`${BASE_URL}/rl-status`);
  if (!res.ok) throw new Error(`RL status error: ${res.statusText}`);
  return res.json();
}
