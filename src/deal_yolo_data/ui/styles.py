import streamlit as st

def inject_style():
    st.markdown(
        """
<style>
:root {
  --bg: #f5f7fb;
  --bg-2: #eef2f7;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;
  --accent-2: #60a5fa;
  --accent-3: #38bdf8;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
}

.stApp {
  background:
    radial-gradient(900px circle at 15% 10%, rgba(96, 165, 250, 0.2), transparent 55%),
    radial-gradient(800px circle at 85% 0%, rgba(56, 189, 248, 0.18), transparent 55%),
    linear-gradient(180deg, #f6f8fc 0%, #eef2f7 45%, #f5f7fb 100%);
  color: var(--text);
  font-family: "Avenir Next", "Source Sans Pro", "Noto Sans", sans-serif;
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(59, 130, 246, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(59, 130, 246, 0.08) 1px, transparent 1px);
  background-size: 36px 36px;
  pointer-events: none;
  z-index: 0;
}

.hero-title {
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  background: linear-gradient(90deg, #3b82f6, #60a5fa, #2563eb, #3b82f6);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: titleGlow 6s linear infinite;
  text-shadow: 0 0 18px rgba(59, 130, 246, 0.25);
}

@keyframes titleGlow {
  0% { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}

.glow-frame {
  position: relative;
  border-radius: 16px;
  padding: 2px;
  background: linear-gradient(120deg, rgba(59, 130, 246, 0.18), rgba(37, 99, 235, 0.25), rgba(59, 130, 246, 0.18));
  background-size: 200% 200%;
  animation: borderFlow 8s ease infinite;
}

.glow-frame > .glow-inner {
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.92);
  padding: 14px 16px;
}

@keyframes borderFlow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.busy-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  color: #3b82f6;
}

.busy-dots span {
  display: inline-block;
  width: 6px;
  height: 6px;
  margin-left: 3px;
  border-radius: 50%;
  background: #3b82f6;
  animation: pulse 1.2s infinite ease-in-out;
}

.busy-dots span:nth-child(2) { animation-delay: 0.2s; }
.busy-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes pulse {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
  border-right: 1px solid var(--border);
}

.sidebar-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.5rem;
}

.panel {
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.12), #ffffff);
  border: 1px solid rgba(37, 99, 235, 0.15);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(6px);
}

.kpi {
  font-size: 0.85rem;
  color: var(--muted);
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.chip-wait {
  background: #f1f5f9;
  color: #64748b;
  border-color: #e2e8f0;
}

.chip-done {
  background: #dcfce7;
  color: #166534;
  border-color: #86efac;
}

.chip-skip {
  background: #fef3c7;
  color: #92400e;
  border-color: #fde68a;
}

.file-card {
  background: #f8fafc;
  border: 1px solid rgba(37, 99, 235, 0.15);
  border-radius: 12px;
  padding: 10px 12px;
  min-height: 70px;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.06);
}

.file-name {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--text);
  word-break: break-all;
}

.file-meta {
  font-size: 0.75rem;
  color: var(--muted);
}

button[kind="primary"] {
  background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%);
  border: none;
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25);
}

div[data-baseweb="input"] > div,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #0f172a !important;
  border-color: rgba(37, 99, 235, 0.2) !important;
}

div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
  color: rgba(100, 116, 139, 0.8) !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid rgba(37, 99, 235, 0.18);
  border-radius: 12px;
  overflow: hidden;
}

div[data-testid="stExpander"] {
  border-radius: 14px;
  border: 1px solid rgba(37, 99, 235, 0.12);
  background: #ffffff;
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(6px);
}

div[data-testid="stExpander"] > details > summary {
  padding: 0.4rem 1rem;
  font-weight: 600;
  color: var(--text);
}

.stepper {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.step {
  display: flex;
  align-items: center;
  gap: 8px;
}

.step-circle {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  border: 1px solid rgba(37, 99, 235, 0.2);
  background: #ffffff;
  color: #64748b;
}

.step-circle.done {
  background: rgba(34, 197, 94, 0.2);
  color: #86efac;
  border-color: rgba(34, 197, 94, 0.5);
}

.step-circle.active {
  background: #dbeafe;
  color: #1d4ed8;
  border-color: #93c5fd;
}

.step-circle.locked {
  background: #f1f5f9;
  color: #94a3b8;
}

.step-circle.skipped {
  background: #fef3c7;
  color: #92400e;
  border-color: #fde68a;
}

.step-label {
  font-size: 0.8rem;
  color: var(--text);
  font-weight: 600;
}

.step-line {
  flex: 1;
  height: 2px;
  min-width: 24px;
  background: #e2e8f0;
}

.step-line.line-done {
  background: linear-gradient(90deg, #22c55e, #16a34a);
}

.step-line.line-skip {
  background: repeating-linear-gradient(90deg, #f59e0b 0 6px, #fde68a 6px 12px);
}

.step-line.line-lock {
  background: #e2e8f0;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 8px;
}

.stat-card {
  background: #ffffff;
  border: 1px solid rgba(37, 99, 235, 0.12);
  border-radius: 12px;
  padding: 12px;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.05);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--muted);
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 4px;
  color: var(--text);
}

.stat-hint {
  font-size: 0.7rem;
  color: var(--muted);
  margin-top: 2px;
}

.dependency-card {
  background: #ffffff;
  border: 1px solid rgba(37, 99, 235, 0.12);
  border-radius: 14px;
  padding: 12px 16px;
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(6px);
}

.file-manager {
  border: 1px solid rgba(37, 99, 235, 0.15);
  border-radius: 14px;
  padding: 12px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.05);
}

.fm-node {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 8px;
}

.fm-selected {
  background: rgba(37, 99, 235, 0.12);
  border: 1px solid rgba(37, 99, 235, 0.3);
}

.fm-node:hover {
  background: rgba(59, 130, 246, 0.08);
}

.fm-icon {
  width: 18px;
  height: 14px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.fm-icon svg {
  width: 18px;
  height: 14px;
  fill: currentColor;
  color: #2563eb;
}

.fm-depth-0 .fm-icon svg { color: #0284c7; }
.fm-depth-1 .fm-icon svg { color: #2563eb; }
.fm-depth-2 .fm-icon svg { color: #7c3aed; }
.fm-depth-3 .fm-icon svg { color: #059669; }
.fm-depth-4 .fm-icon svg { color: #f59e0b; }

.fm-name {
  font-size: 0.82rem;
  color: #0f172a;
  font-weight: 600;
}

.fm-path {
  font-size: 0.75rem;
  color: #64748b;
}

.drop-zone {
  border: 1px dashed rgba(37, 99, 235, 0.4);
  border-radius: 12px;
  padding: 10px;
  text-align: center;
  color: #2563eb;
  background: rgba(59, 130, 246, 0.05);
  font-size: 0.8rem;
}

hr {
  border: none;
  border-top: 1px solid rgba(37, 99, 235, 0.12);
}
</style>
        """,
        unsafe_allow_html=True,
    )
