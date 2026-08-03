#!/usr/bin/env python3
"""Streamlit front end for the flash-calculation model ensemble.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import inference as inf

st.set_page_config(page_title="Flash Calculation Predictor", page_icon="\U0001F9EA",
                   layout="wide")

# Purple theme palette
PURPLE = "#7A3FB8"
PURPLE_DARK = "#4A1E7A"
PURPLE_LIGHT = "#B98BE0"
TEAL = "#17A398"
GOLD = "#E8A32C"
RED = "#D64550"
PLOT_BG = "rgba(0,0,0,0)"
GRID = "rgba(122,63,184,0.15)"

# Max points actually drawn in a plot (subsample above this for fast rendering)
MAX_PLOT_POINTS = 2500

Y_COLUMNS = inf.Y_COLUMNS
COMP_LABELS = ["C1 (1)", "C3 (3)", "C14 (14)"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), engine="python", on_bad_lines="skip")


@st.cache_data(show_spinner="Running the 7-fold ensemble...")
def _run_prediction(model_key: str, df: pd.DataFrame, device_str: str,
                    component_set: str) -> tuple:
    """Cached prediction pipeline.

    Returns (y_pred, y_std, X_scaled, has_gt, metrics, y_true, elapsed_s).
    """
    t0 = time.perf_counter()
    y_pred, y_std, X_scaled, has_gt = inf.predict(model_key, df, device_str)
    elapsed = time.perf_counter() - t0
    y_true = df[Y_COLUMNS].to_numpy(dtype=float) if has_gt else None
    metrics = inf.compute_metrics(y_pred, X_scaled, has_gt, y_true,
                                  component_set=component_set)
    return y_pred, y_std, X_scaled, has_gt, metrics, y_true, elapsed


def _subsample_idx(n: int, cap: int = MAX_PLOT_POINTS) -> np.ndarray:
    """Deterministic even subsample of row indices for fast plotting."""
    if n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).astype(int)


@st.cache_data(show_spinner=False)
def _ternary(comps: np.ndarray, which: str) -> go.Figure:
    """Ternary scatter of predicted liquid (x) or vapor (y) compositions.

    ``comps`` is an [N, 3] array of (C1, C3, C14) fractions.
    """
    if which == "liquid":
        title, color = "Predicted liquid composition  (x)", PURPLE
    else:
        title, color = "Predicted vapor composition  (y)", TEAL

    idx = _subsample_idx(len(comps))
    a, b, c = comps[idx, 0], comps[idx, 1], comps[idx, 2]
    fig = go.Figure(go.Scatterternary(
        a=a, b=b, c=c, mode="markers",
        marker={"size": 6, "color": color, "opacity": 0.55,
                "line": {"width": 0.4, "color": "white"}},
        hovertemplate="C1=%{a:.3f}<br>C3=%{b:.3f}<br>C14=%{c:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": PURPLE_DARK}},
        ternary={"sum": 1, "bgcolor": "rgba(122,63,184,0.04)",
                 "aaxis": {"title": "C1", "min": 0.0, "gridcolor": GRID, "linecolor": PURPLE_LIGHT},
                 "baxis": {"title": "C3", "min": 0.0, "gridcolor": GRID, "linecolor": PURPLE_LIGHT},
                 "caxis": {"title": "C14", "min": 0.0, "gridcolor": GRID, "linecolor": PURPLE_LIGHT}},
        paper_bgcolor=PLOT_BG, height=440, margin={"l": 40, "r": 40, "t": 55, "b": 40},
    )
    if len(comps) > MAX_PLOT_POINTS:
        fig.add_annotation(text=f"showing {MAX_PLOT_POINTS:,} of {len(comps):,} points",
                           showarrow=False, x=0.5, y=-0.08, xref="paper", yref="paper",
                           font={"size": 10, "color": "#888"})
    return fig


@st.cache_data(show_spinner=False)
def _parity(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """7-panel predicted-vs-true parity plot (WebGL, subsampled)."""
    titles = ["x1", "x3", "x14", "y1", "y3", "y14", "L"]
    idx = _subsample_idx(len(y_true))
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.14)
    for k in range(7):
        r, c = divmod(k, 4)
        t, p = y_true[idx, k], y_pred[idx, k]
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        r2 = _r2(y_true[:, k], y_pred[:, k])
        fig.add_trace(go.Scattergl(x=[lo, hi], y=[lo, hi], mode="lines",
                                   line={"color": RED, "dash": "dash", "width": 1.4},
                                   hoverinfo="skip", showlegend=False), row=r + 1, col=c + 1)
        fig.add_trace(go.Scattergl(x=t, y=p, mode="markers",
                                   marker={"size": 4, "color": PURPLE, "opacity": 0.45},
                                   hovertemplate="true=%{x:.3f}<br>pred=%{y:.3f}<extra></extra>",
                                   showlegend=False), row=r + 1, col=c + 1)
        ax = "" if k == 0 else str(k + 1)
        fig.add_annotation(text=f"R\u00b2={r2:.3f}", showarrow=False,
                           xref=f"x{ax} domain", yref=f"y{ax} domain",
                           x=0.05, y=0.95, align="left",
                           font={"size": 11, "color": PURPLE_DARK})
    fig.update_layout(height=560, margin={"l": 45, "r": 20, "t": 55, "b": 45},
                      paper_bgcolor=PLOT_BG, plot_bgcolor="rgba(122,63,184,0.03)",
                      title={"text": "Predicted vs True  (parity)",
                             "font": {"size": 16, "color": PURPLE_DARK}})
    fig.update_xaxes(title_text="true", row=2, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text="pred", col=1, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _uncertainty_bar(y_std: np.ndarray) -> go.Figure:
    """Mean per-output ensemble spread (fold std) as a horizontal bar chart."""
    labels = Y_COLUMNS
    means = y_std.mean(axis=0)
    fig = go.Figure(go.Bar(
        x=means, y=labels, orientation="h",
        marker={"color": means, "colorscale": [[0, PURPLE_LIGHT], [1, PURPLE_DARK]]},
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": "Ensemble uncertainty  (mean fold spread)",
               "font": {"size": 16, "color": PURPLE_DARK}},
        height=440, margin={"l": 50, "r": 30, "t": 55, "b": 40},
        paper_bgcolor=PLOT_BG, plot_bgcolor="rgba(122,63,184,0.03)",
        xaxis={"title": "std. deviation across folds", "gridcolor": GRID},
        yaxis={"autorange": "reversed"},
    )
    return fig


def _r2(t: np.ndarray, p: np.ndarray) -> float:
    sse = float(np.sum((t - p) ** 2))
    sst = float(np.sum((t - np.mean(t)) ** 2)) + 1e-12
    return 1.0 - sse / sst


def _metric_card(label: str, value, sub: str = "") -> str:
    txt = value if isinstance(value, str) else f"{value:.5f}"
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{txt}</div>"
        f"<div class='metric-sub'>{sub}</div>"
        f"</div>"
    )


# -----------------------------------------------------------------------------
# Global styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#FBFAFF 0%,#F3EEFB 100%); }
      section[data-testid="stSidebar"] {
          background: linear-gradient(180deg,#4A1E7A 0%,#7A3FB8 100%);
      }
      section[data-testid="stSidebar"] * { color: #F3EEFB !important; }
      section[data-testid="stSidebar"] code {
          background: rgba(255,255,255,0.16) !important;
          color: #FDE9A9 !important; padding: 0 4px; border-radius: 4px;
      }
      section[data-testid="stSidebar"] div[data-baseweb="select"] * {
          color: #241033 !important;
      }
      div[data-baseweb="popover"] li, div[data-baseweb="menu"] * {
          color: #241033 !important;
      }
      .hero {
          background: linear-gradient(120deg,#4A1E7A 0%,#7A3FB8 55%,#B98BE0 100%);
          padding: 1.4rem 1.8rem; border-radius: 16px; color: #fff;
          box-shadow: 0 6px 22px rgba(74,30,122,0.28); margin-bottom: 1.2rem;
      }
      .hero h1 { margin: 0; font-size: 2.0rem; font-weight: 800; letter-spacing:-0.5px; }
      .hero p  { margin: 0.4rem 0 0 0; font-size: 0.98rem; opacity: 0.92; }
      .section-title {
          color:#4A1E7A; font-size:1.35rem; font-weight:750;
          border-left:5px solid #7A3FB8; padding-left:0.6rem; margin:0.6rem 0 0.4rem 0;
      }
      .metric-card {
          background:#fff; border-radius:14px; padding:1rem 1.1rem;
          border:1px solid #E6DAF7; box-shadow:0 3px 12px rgba(74,30,122,0.08);
          height:100%;
      }
      .metric-label { color:#7A3FB8; font-size:0.82rem; font-weight:700;
                      text-transform:uppercase; letter-spacing:0.5px; }
      .metric-value { color:#241033; font-size:1.7rem; font-weight:800; line-height:1.2; }
      .metric-sub   { color:#8A7BA0; font-size:0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("## \u2699\ufe0f Configuration")
models = inf.available_models()
model_key = st.sidebar.selectbox("Model / training scheme", models,
                                 index=models.index("Modified LR annealing (beta=2.2)")
                                 if "Modified LR annealing (beta=2.2)" in models else 0)
st.sidebar.caption("Predictions are the mean of the model's 7 cross-validation folds; "
                   "the per-fold spread is reported as uncertainty.")
st.sidebar.markdown("---")
component_set = st.sidebar.selectbox(
    "Fugacity metric: component set", list(inf.COMPONENT_SETS),
    index=list(inf.COMPONENT_SETS).index(inf.DEFAULT_COMPONENT_SET))
st.sidebar.caption("Sets the Peng-Robinson constants used for the iso-fugacity "
                   "residual. Choose the components your uploaded file actually "
                   "represents \u2014 otherwise FugRMSE evaluates the wrong "
                   "thermodynamics.")
st.sidebar.markdown("---")
st.sidebar.markdown("### Expected columns")
st.sidebar.markdown(
    "**Required:** `z1, z3, z14, p` + phase info (`L`, *or* `Gas` & `2phase`).\n\n"
    "**Optional (enables R\u00b2):** true `x1, x3, x14, y1, y3, y14`."
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><h1>\U0001F9EA Flash Calculation Predictor</h1>"
    "<p>Upload feed data to predict the equilibrium phase split "
    "(<b>x</b>, <b>y</b>, <b>L</b>) with a neural model and get "
    "physics-consistency metrics. Include ground-truth compositions to also "
    "score accuracy (<b>R\u00b2</b>).</p></div>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

with st.expander("Don't have a file? Generate a sample from the reference set"):
    n_sample = st.number_input("Rows", min_value=5, max_value=5000, value=200, step=5)
    include_gt = st.checkbox("Include ground-truth columns (x, y)", value=True)
    if st.button("Build sample CSV"):
        ref = pd.read_csv(inf.REFERENCE_CSV, engine="python", on_bad_lines="skip")
        keep = ["z1", "z3", "z14", "p", "L"] + (inf.GT_COLUMNS if include_gt else [])
        sample = ref[keep].sample(n=int(n_sample), random_state=0).reset_index(drop=True)
        st.session_state["sample_csv"] = sample.to_csv(index=False).encode()
        st.success(f"Built a {int(n_sample)}-row sample.")
    if "sample_csv" in st.session_state:
        st.download_button("Download sample CSV", st.session_state["sample_csv"],
                           file_name="sample_input.csv", mime="text/csv")

if uploaded is None:
    st.info("Awaiting a CSV upload to run predictions.")
    st.stop()

df_in = _read_csv(uploaded.getvalue())

with st.expander(f"Input preview  ({len(df_in):,} rows)", expanded=False):
    st.dataframe(df_in.head(20), use_container_width=True)

try:
    y_pred, y_std, X_scaled, has_gt, metrics, y_true, elapsed = _run_prediction(
        model_key, df_in, "cpu", component_set)
    z_recon = inf.reconstruct_z(y_pred, X_scaled)
    out_df = inf.predictions_dataframe(df_in, y_pred, y_std)
    out_df[["z1_recon", "z3_recon", "z14_recon"]] = z_recon
except (ValueError, KeyError, FileNotFoundError) as exc:
    st.error(f"Could not run predictions: {exc}")
    st.stop()

n_rows = len(df_in)
per_row_ms = (elapsed / n_rows * 1000.0) if n_rows else 0.0
st.markdown(
    f"<div style='background:#fff;border:1px solid #E6DAF7;border-radius:12px;"
    f"padding:0.7rem 1rem;margin:0.3rem 0 0.8rem 0;box-shadow:0 3px 12px "
    f"rgba(74,30,122,0.08);color:#4A1E7A;font-weight:600;'>"
    f"\u26A1 Predicted <b>{n_rows:,}</b> rows with <b>{model_key}</b> "
    f"in <b>{elapsed*1000:.1f} ms</b> "
    f"<span style='color:#8A7BA0;font-weight:500;'>"
    f"({per_row_ms:.3f} ms/row, 7-fold ensemble)</span></div>",
    unsafe_allow_html=True,
)

# Ground-truth banner
if has_gt:
    st.success("Ground-truth compositions detected \u2014 reporting **R\u00b2** plus the "
               "three residuals (fugacity gating uses the true labels).")
else:
    st.info("No ground-truth compositions \u2014 reporting the three physics "
            "residuals only (fugacity gating uses the model's predictions). "
            "R\u00b2 is unavailable without labels.")

# Metrics row
st.markdown("<div class='section-title'>Metrics</div>", unsafe_allow_html=True)
cards = st.columns(4)
r2_txt = f"{metrics['R2']:.4f}" if has_gt else "n/a"
cards[0].markdown(_metric_card("R\u00b2", r2_txt,
                               "needs ground truth"), unsafe_allow_html=True)
cards[1].markdown(_metric_card("FugRMSE", metrics["FugRMSE"],
                               f"{metrics['fug_components']}"), unsafe_allow_html=True)
cards[2].markdown(_metric_card("CompRMSE", metrics["CompRMSE"],
                               "composition closure"), unsafe_allow_html=True)
cards[3].markdown(_metric_card("InterRMSE", metrics["InterRMSE"],
                               "mass-balance closure"), unsafe_allow_html=True)
# Plots
st.markdown("<div class='section-title'>Visualizations</div>", unsafe_allow_html=True)
liq = out_df[["x1_pred", "x3_pred", "x14_pred"]].to_numpy(dtype=float)
vap = out_df[["y1_pred", "y3_pred", "y14_pred"]].to_numpy(dtype=float)
if has_gt:
    tab_tern, tab_par, tab_unc = st.tabs(
        ["Ternary (predicted)", "Parity (pred vs true)", "Uncertainty"])
    with tab_tern:
        c1, c2 = st.columns(2)
        c1.plotly_chart(_ternary(liq, "liquid"), use_container_width=True)
        c2.plotly_chart(_ternary(vap, "vapor"), use_container_width=True)
    with tab_par:
        st.plotly_chart(_parity(y_true, y_pred), use_container_width=True)
    with tab_unc:
        st.plotly_chart(_uncertainty_bar(y_std), use_container_width=True)
else:
    tab_tern, tab_unc = st.tabs(["Ternary (predicted)", "Uncertainty"])
    with tab_tern:
        c1, c2 = st.columns(2)
        c1.plotly_chart(_ternary(liq, "liquid"), use_container_width=True)
        c2.plotly_chart(_ternary(vap, "vapor"), use_container_width=True)
    with tab_unc:
        st.plotly_chart(_uncertainty_bar(y_std), use_container_width=True)

# Results table + download
st.markdown("<div class='section-title'>Predictions</div>", unsafe_allow_html=True)
opt1, opt2 = st.columns(2)
show_table = opt1.checkbox("Show predictions table", value=True)
show_download = opt2.checkbox("Show download button", value=True)
if show_table:
    st.dataframe(out_df.head(1000), use_container_width=True, height=360)
    if len(out_df) > 1000:
        st.caption(f"Showing first 1,000 of {len(out_df):,} rows "
                   "(download for the full table).")
if show_download:
    st.download_button("Download predictions CSV",
                       out_df.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_key.replace(' ', '_')}.csv",
                       mime="text/csv")
