document.addEventListener('DOMContentLoaded', () => {
  const queryInput  = document.getElementById('queryInput');
  const searchBtn   = document.getElementById('searchBtn');
  const loading     = document.getElementById('loading');
  const results     = document.getElementById('results');

  // Sample query chips
  document.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.dataset.query;
      runSearch();
    });
  });

  // Search triggers
  searchBtn.addEventListener('click', runSearch);
  queryInput.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });

  // Auto-run default query on load
  queryInput.value = 'What is the critical transition temperature of YBCO?';
  runSearch();

  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`pane-${tab.dataset.tab}`).classList.add('active');
    });
  });

  // ── Core search ──────────────────────────────────────────────────
  async function runSearch() {
    const q = queryInput.value.trim();
    if (!q) return;

    loading.classList.remove('hidden');
    results.classList.add('hidden');

    try {
      const res  = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      });
      if (!res.ok) throw new Error('Server error');
      const data = await res.json();
      render(data);
      results.classList.remove('hidden');
    } catch (err) {
      alert('Query failed: ' + err.message);
    } finally {
      loading.classList.add('hidden');
    }
  }

  // ── Render ───────────────────────────────────────────────────────
  function render(data) {
    const { entity_linking: el, divergence_analysis: da, retrieval_comparison: rc, synthesized_answer: sa } = data;

    // Entity Resolution
    set('r-surface', el.detected_surface_form || '—');
    set('r-id',      el.canonical_id || 'NIL');
    set('r-name',    el.primary_name || 'Unresolved');
    set('r-conf-text', `${el.confidence.toFixed(3)}  (τ = ${el.threshold_tau})`);
    document.getElementById('r-conf-fill').style.width = `${Math.min(100, el.confidence * 100)}%`;

    const aliasBox = document.getElementById('r-aliases');
    aliasBox.innerHTML = '';
    (el.known_aliases || []).slice(0, 8).forEach(a => {
      const s = document.createElement('span');
      s.className = 'alias-pill';
      s.textContent = a;
      aliasBox.appendChild(s);
    });

    // Divergence
    set('r-lexsim', da.lexical_similarity.toFixed(4));
    const bucketEl = document.getElementById('r-bucket');
    bucketEl.textContent = da.bucket;
    bucketEl.className = 'info-val ' + bucketClass(da.bucket);
    document.getElementById('r-bucket-desc').textContent = bucketDescription(da.bucket);

    // Best answer
    set('r-answer', sa || 'No evidence retrieved.');

    // Passage lists
    renderPassages('pane-cond5', rc.condition_5_full_system);
    renderPassages('pane-cond3', rc.condition_3_baseline);
    renderPassages('pane-cond4', rc.condition_4_entity_only);
  }

  function renderPassages(paneId, passages) {
    const pane = document.getElementById(paneId);
    const box  = pane.querySelector('.passages');
    box.innerHTML = '';

    if (!passages || passages.length === 0) {
      box.innerHTML = '<p class="no-results">No passages retrieved for this condition.</p>';
      return;
    }

    passages.forEach(p => {
      const div = document.createElement('div');
      div.className = 'passage-item';
      div.innerHTML = `
        <div class="passage-meta">
          <span class="passage-rank">Rank #${p.rank}</span>
          <span class="passage-entity">${esc(p.canonical_entity)}</span>
          <span class="passage-score">score ${p.score.toFixed(4)}</span>
        </div>
        <div class="passage-title">${esc(p.title)}</div>
        <div class="passage-text">${esc(p.text)}</div>
      `;
      box.appendChild(div);
    });
  }

  // ── Helpers ──────────────────────────────────────────────────────
  function set(id, text) {
    document.getElementById(id).textContent = text;
  }

  function esc(str) {
    if (!str) return '';
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function bucketClass(bucket) {
    if (bucket === 'Full alias-mismatch') return 'bucket-full';
    if (bucket === 'Partial-mismatch')    return 'bucket-partial';
    return 'bucket-none';
  }

  function bucketDescription(bucket) {
    if (bucket === 'Full alias-mismatch')
      return 'Query and evidence use entirely different names for the same material. Entity linking maps the alias to a canonical ID to recover relevant passages.';
    if (bucket === 'Partial-mismatch')
      return 'Some terminology overlap exists, but the query and evidence use different surface forms. Entity linking helps bridge the gap.';
    return 'Query and evidence share significant lexical overlap. Standard BM25/Dense retrieval is sufficient.';
  }
});
