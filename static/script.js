document.addEventListener('DOMContentLoaded', () => {
  fetchModelInfo();
  handleTokenizeInput();
});

function toggleTheme() {
  document.body.classList.toggle('light-theme');
}

function switchNavTab(tabId) {
  document.querySelectorAll('.view-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');

  if (tabId === 'generate-view') document.getElementById('tab-btn-generate').classList.add('active');
  if (tabId === 'tokenizer-view') document.getElementById('tab-btn-tokenizer').classList.add('active');
  if (tabId === 'training-view') document.getElementById('tab-btn-training').classList.add('active');
  if (tabId === 'scaling-view') document.getElementById('tab-btn-scaling').classList.add('active');
}

function updateSamplingValues() {
  document.getElementById('temp-val').innerText = document.getElementById('temp-slider').value;
  document.getElementById('topk-val').innerText = document.getElementById('topk-slider').value;
  document.getElementById('topp-val').innerText = document.getElementById('topp-slider').value;
  document.getElementById('tokens-val').innerText = document.getElementById('tokens-slider').value;
}

// 1. Text Generation
async function generateText() {
  const prompt = document.getElementById('prompt-input').value.trim();
  if (!prompt) return;

  const btn = document.getElementById('gen-btn');
  const outputBox = document.getElementById('output-box');
  const badge = document.getElementById('gen-time-badge');

  btn.disabled = true;
  btn.innerText = '⏳ Sampling tokens...';
  outputBox.innerHTML = '<span style="color:var(--text-muted)">Generating autoregressive continuation...</span>';

  const payload = {
    prompt: prompt,
    temperature: parseFloat(document.getElementById('temp-slider').value),
    top_k: parseInt(document.getElementById('topk-slider').value, 10),
    top_p: parseFloat(document.getElementById('topp-slider').value),
    max_new_tokens: parseInt(document.getElementById('tokens-slider').value, 10)
  };

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await resp.json();
    outputBox.innerText = data.generated_text;
    badge.innerText = `${data.generation_time_ms} ms`;
    badge.style.display = 'inline';
  } catch (e) {
    console.error('Gen error:', e);
    outputBox.innerHTML = '<span style="color:red">Error generating text.</span>';
  } finally {
    btn.disabled = false;
    btn.innerText = '🚀 Generate Text Continuation';
  }
}

// 2. Tokenizer Visualizer
let tokenizeTimeout = null;
function handleTokenizeInput() {
  const text = document.getElementById('token-input').value;
  if (tokenizeTimeout) clearTimeout(tokenizeTimeout);

  tokenizeTimeout = setTimeout(async () => {
    try {
      const resp = await fetch('/api/tokenize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
      });

      const data = await resp.json();
      document.getElementById('token-count').innerText = data.token_count;

      const chips = document.getElementById('tokens-chips');
      chips.innerHTML = data.tokens.map((tok, idx) => `
        <div class="token-chip">
          <span>${escapeHtml(tok)}</span>
          <span class="token-id">#${data.token_ids[idx]}</span>
        </div>
      `).join('');
    } catch (e) {
      console.error('Tokenize error:', e);
    }
  }, 150);
}

// 3. Training Step
async function runTrainStep(mode) {
  const statusBox = document.getElementById('train-status');
  statusBox.style.display = 'block';
  statusBox.innerHTML = `⏳ Executing ${mode.toUpperCase()} training epoch...`;

  try {
    const resp = await fetch('/api/train_step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: mode, epochs: 1 })
    });

    const data = await resp.json();
    const latest = data.new_metrics[data.new_metrics.length - 1];

    document.getElementById('stat-loss').innerText = latest.train_loss;
    document.getElementById('stat-ppl').innerText = latest.train_perplexity;
    document.getElementById('stat-epochs').innerText = data.total_epochs_run;

    statusBox.innerHTML = `<span style="color:#10b981">✅ ${mode.toUpperCase()} epoch complete! Loss: ${latest.train_loss}, Perplexity: ${latest.train_perplexity}</span>`;

    // Append to table
    const tbody = document.querySelector('#train-history-table tbody');
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><strong>Epoch ${data.total_epochs_run}</strong></td>
      <td><code>${mode}</code></td>
      <td>${latest.train_loss}</td>
      <td><code>${latest.train_perplexity}</code></td>
      <td>${latest.epoch_time_sec}s</td>
    `;
    tbody.appendChild(row);
  } catch (e) {
    console.error('Train error:', e);
    statusBox.innerHTML = '<span style="color:#ef4444">❌ Training error.</span>';
  }
}

// 4. Model Info Scaling
async function fetchModelInfo() {
  try {
    const resp = await fetch('/api/model_info');
    const data = await resp.json();

    document.getElementById('scale-params').innerText = `${data.num_params.toLocaleString()} (${data.num_params_m} M)`;
    document.getElementById('scale-dmodel').innerText = data.d_model;
    document.getElementById('scale-layers').innerText = data.n_layer;
    document.getElementById('scale-seqlen').innerText = data.max_seq_len;
    document.getElementById('scale-vocab').innerText = data.vocab_size;
  } catch (e) {
    console.error('Info error:', e);
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
