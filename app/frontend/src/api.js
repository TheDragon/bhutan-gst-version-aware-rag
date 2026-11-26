const API_ROOT = import.meta.env.VITE_API_URL || ''

async function request(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

export const api = {
  health: () => request('/api/health'),
  stats: () => request('/api/stats'),
  documents: () => request('/api/documents'),
  timeline: (section) => request(`/api/timeline/${encodeURIComponent(section)}`),
  amendments: () => request('/api/amendments'),
  provision: (instrument, section) =>
    request(`/api/provision?instrument=${encodeURIComponent(instrument)}&section=${encodeURIComponent(section)}`),
  answer: (payload) => request('/api/answer', { method: 'POST', body: JSON.stringify(payload) }),
  search: (payload) => request('/api/search', { method: 'POST', body: JSON.stringify(payload) }),
  compare: (payload) => request('/api/compare', { method: 'POST', body: JSON.stringify(payload) }),
}
