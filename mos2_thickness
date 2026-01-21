# app.py
# Streamlit web app: MoS2 thickness estimation from absorption spectrum (TXT upload)
# Revised version:
# - Segment-based prediction (thin / transition / thick)
# - Robust dip detection (shoulders OUTSIDE dip window)
# - For thick films (dip detected): output thickness from dip position & dip absorptance
# - Always output thickness from A peak and B peak for reference
#
# Run:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run app.py

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ========= Calibration constants (from the paper) =========
# A exciton: alpha = 0.9694e6 cm^-1, m = 75.93%
# B exciton: alpha = 0.9516e6 cm^-1, m = 79.45%
ALPHA_A = 0.9694e6  # cm^-1
M_A = 0.7593        # fraction (0-1)
ALPHA_B = 0.9516e6  # cm^-1
M_B = 0.7945        # fraction (0-1)
CM_TO_NM = 1e7

# Default windows (nm)
DEFAULT_A_WIN = (650.0, 680.0)    # A exciton peak window
DEFAULT_B_WIN = (595.0, 625.0)    # B exciton peak window
DEFAULT_DIP_WIN = (685.0, 710.0)  # dip window (thick-film indicator)


# ========= Utility functions =========
def load_txt_as_df(uploaded_file) -> pd.DataFrame:
    """
    Read TXT with flexible separators:
      - whitespace/tab/comma/semicolon
      - header optional
      - ignores empty lines and lines starting with '#'
    """
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("文件为空或只有注释行（以 # 开头）。")

    first = lines[0]
    has_header = bool(re.search(r"[A-Za-z]", first))

    buf = io.StringIO("\n".join(lines))
    df = pd.read_csv(
        buf,
        sep=r"[\s,;]+",
        engine="python",
        header=0 if has_header else None,
    )

    if df.shape[1] < 2:
        raise ValueError("未解析出至少两列数据（波长、吸收率）。请检查分隔符或格式。")

    if not has_header:
        df = df.rename(columns={df.columns[0]: "wavelength_nm", df.columns[1]: "absorptance"})
    return df


def to_fraction(y: np.ndarray) -> np.ndarray:
    """Auto convert absorptance to 0-1 fraction if data looks like percent."""
    y = y.astype(float)
    if np.nanmax(y) > 1.5:
        return y / 100.0
    return y


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average smoothing (no scipy)."""
    win = int(win)
    if win <= 1:
        return y.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(ypad, kernel, mode="valid")


def find_peak_in_window(x, y, lo, hi):
    """Return (peak_wavelength, peak_value) for max within [lo, hi]."""
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        return None, None
    xs, ys = x[mask], y[mask]
    idx = int(np.nanargmax(ys))
    return float(xs[idx]), float(ys[idx])


def find_min_in_window(x, y, lo, hi):
    """Return (min_wavelength, min_value) for min within [lo, hi]."""
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        return None, None
    xs, ys = x[mask], y[mask]
    idx = int(np.nanargmin(ys))
    return float(xs[idx]), float(ys[idx])


def thickness_from_exciton(A: float, alpha: float, m: float) -> float:
    """
    Invert A = m(1-exp(-alpha*d)) -> d (nm).
    A and m are fractions (0-1).
    """
    eps = 1e-9
    if A is None or not np.isfinite(A) or A <= 0:
        return float("nan")
    if A >= m:
        A = m - eps
    d_cm = -(1.0 / alpha) * np.log(1.0 - A / m)
    return d_cm * CM_TO_NM


# ===== dip-based thickness (empirical equations in the paper) =====
def thickness_from_dip_position(dip_pos_nm: float) -> float:
    # y1 = 0.3432 d + 672  => d = (y1 - 672)/0.3432
    return (dip_pos_nm - 672.0) / 0.3432


def thickness_from_dip_absorptance(dip_abs_fraction: float) -> float:
    # y2 = -1.018 d + 100%  => d = (100 - y2)/1.018
    dip_abs_pct = dip_abs_fraction * 100.0
    return (100.0 - dip_abs_pct) / 1.018


def detect_dip(x, y, dip_lo, dip_hi, shoulder_width_nm=8.0, min_drop=0.010):
    """
    Robust dip detection:
      - find minimum in [dip_lo, dip_hi]
      - compute shoulders average from OUTSIDE the window:
          left:  [dip_lo - shoulder_width_nm, dip_lo]
          right: [dip_hi, dip_hi + shoulder_width_nm]
      - dip exists if (shoulders_avg - dip_val) >= min_drop
    y is fraction (0-1).
    """
    dip_pos, dip_val = find_min_in_window(x, y, dip_lo, dip_hi)
    if dip_pos is None:
        return False, None, None, None, None

    left_mask = (x >= dip_lo - shoulder_width_nm) & (x <= dip_lo)
    right_mask = (x >= dip_hi) & (x <= dip_hi + shoulder_width_nm)

    if not (np.any(left_mask) and np.any(right_mask)):
        # can't judge significance well; still return the dip candidate
        return False, float(dip_pos), float(dip_val), None, None

    shoulders_avg = float(np.nanmean(np.concatenate([y[left_mask], y[right_mask]])))
    drop = shoulders_avg - float(dip_val)
    return (drop >= min_drop), float(dip_pos), float(dip_val), float(drop), float(shoulders_avg)


# ========= Streamlit UI =========
st.set_page_config(page_title="MoS₂ Thickness (Segmented, Dip Output)", layout="centered")
st.title("MoS₂ 薄膜厚度推断")

st.markdown(
    """
上传 **txt** 吸收光谱（两列：波长 nm、吸收率），程序将：
- 绘制谱图（可选平滑）
- 提取 **A 峰 (~660 nm)**、**B 峰 (~610 nm)** 并分别反演厚度
- 检测 **~700 nm 干涉 dip**（厚膜指示），并给出：
  - dip **位置→厚度**
  - dip **深度→厚度**
- 分段给出“建议厚度”（厚膜段优先 dip）
"""
)

uploaded = st.file_uploader("上传吸收光谱 .txt", type=["txt"])
if uploaded is None:
    st.info("请先上传一个 txt 光谱文件。")
    st.stop()

# ---------- Load ----------
try:
    df = load_txt_as_df(uploaded)
except Exception as e:
    st.error(f"读取失败：{e}")
    st.stop()

st.subheader("数据列选择")
cols = list(df.columns)
c1, c2 = st.columns(2)
with c1:
    wcol = st.selectbox("波长列 (nm)", options=cols, index=0)
with c2:
    acol = st.selectbox("吸收率列", options=cols, index=1 if len(cols) > 1 else 0)

x = pd.to_numeric(df[wcol], errors="coerce").to_numpy(dtype=float)
y_raw = pd.to_numeric(df[acol], errors="coerce").to_numpy(dtype=float)

mask = np.isfinite(x) & np.isfinite(y_raw)
x = x[mask]
y_raw = y_raw[mask]

if x.size < 20:
    st.error("有效数据点太少（<20），请检查文件内容或列选择。")
    st.stop()

# Sort by wavelength
order = np.argsort(x)
x = x[order]
y_raw = y_raw[order]
y = to_fraction(y_raw)

# ---------- Controls ----------
with st.expander("参数设置（建议先用默认值）", expanded=False):
    st.markdown("### 峰/凹陷搜索窗口（nm）")
    colA, colB, colD = st.columns(3)
    with colA:
        a_lo = st.number_input("A峰下限", value=float(DEFAULT_A_WIN[0]), step=1.0)
        a_hi = st.number_input("A峰上限", value=float(DEFAULT_A_WIN[1]), step=1.0)
    with colB:
        b_lo = st.number_input("B峰下限", value=float(DEFAULT_B_WIN[0]), step=1.0)
        b_hi = st.number_input("B峰上限", value=float(DEFAULT_B_WIN[1]), step=1.0)
    with colD:
        d_lo = st.number_input("dip下限", value=float(DEFAULT_DIP_WIN[0]), step=1.0)
        d_hi = st.number_input("dip上限", value=float(DEFAULT_DIP_WIN[1]), step=1.0)

    st.markdown("### 平滑（可选）")
    smooth_on = st.checkbox("启用移动平均平滑", value=True)
    smooth_win = st.slider("平滑窗口点数（奇数更好）", min_value=1, max_value=51, value=9, step=2)

    st.markdown("### dip 显著性判据（厚膜判定）")
    dip_drop = st.number_input("dip 最小显著 drop（0-1）", value=0.010, step=0.001, format="%.3f")
    shoulder_nm = st.number_input("dip 肩部宽度（nm，窗口外）", value=8.0, step=1.0)

    st.markdown("### 分段阈值（无 dip 情况下）")
    thin_threshold = st.number_input("薄膜判定阈值（nm）：mean(dA,dB) < 该值 => thin", value=20.0, step=1.0)

# Use smoothed curve for feature extraction (plot raw + smoothed)
y_feat = moving_average(y, smooth_win) if smooth_on else y.copy()

# ---------- Feature extraction ----------
a_pos, a_abs = find_peak_in_window(x, y_feat, float(a_lo), float(a_hi))
b_pos, b_abs = find_peak_in_window(x, y_feat, float(b_lo), float(b_hi))

tA = thickness_from_exciton(a_abs, ALPHA_A, M_A) if a_abs is not None else float("nan")
tB = thickness_from_exciton(b_abs, ALPHA_B, M_B) if b_abs is not None else float("nan")

dip_detected, dip_pos, dip_val, dip_drop_val, dip_shoulders = detect_dip(
    x, y_feat, float(d_lo), float(d_hi),
    shoulder_width_nm=float(shoulder_nm),
    min_drop=float(dip_drop),
)

tDipPos = float("nan")
tDipAbs = float("nan")
if dip_pos is not None and dip_val is not None:
    # compute dip-based thickness even if not "detected" (for visibility),
    # but we'll only use it as primary when dip_detected==True
    tDipPos = thickness_from_dip_position(dip_pos)
    tDipAbs = thickness_from_dip_absorptance(dip_val)

# ---------- Segmentation & fusion ----------
notes = []
zone = "unknown"
suggested = float("nan")
uncert = float("nan")

validA = np.isfinite(tA)
validB = np.isfinite(tB)

# Primary gating: dip
if dip_detected and np.isfinite(tDipPos) and np.isfinite(tDipAbs):
    zone = "厚膜"
    # thick-film: prefer dip
    suggested = float(np.mean([tDipPos, tDipAbs]))
    uncert = float(abs(tDipPos - tDipAbs) / 2.0)
    notes.append("检测到明显 dip：厚膜段优先采用 dip 反演；A/B 结果仅作对照参考。")
else:
    # No (significant) dip: use A/B
    cand = []
    if validA: cand.append(tA)
    if validB: cand.append(tB)
    if cand:
        mean_ab = float(np.mean(cand))
        if mean_ab < float(thin_threshold):
            zone = "薄膜"
            suggested = mean_ab
            uncert = float(np.std(cand)) if len(cand) >= 2 else float("nan")
            notes.append("未检测到明显 dip 且厚度较薄：使用 A/B 峰强度反演为主。")
        else:
            zone = "可能是厚膜"
            if validA and validB:
                suggested = 0.7 * tA + 0.3 * tB
                uncert = abs(tA - tB) / 2.0
            else:
                suggested = mean_ab
                uncert = float("nan")
            notes.append("未检测到明显 dip，且厚度进入过渡区：A/B 灵敏度下降，建议结合不确定性与本地标定。")
    else:
        zone = "unknown"
        notes.append("A/B 峰不可用：请检查窗口、数据质量、吸收率定义或波段范围。")

# If dip candidate exists but not significant, add note
if (dip_pos is not None) and (not dip_detected):
    if dip_drop_val is None:
        notes.append("发现 dip 候选最小值，但肩部数据不足，无法判断显著性；请扩大波段或调整 shoulder_nm。")
    else:
        notes.append("发现 dip 候选最小值，但显著性不足（drop < 阈值）；可能是噪声或样品未进入厚膜干涉区。")

# ---------- Plot ----------
st.subheader("吸收光谱")
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=1.2, label="raw")
if smooth_on:
    ax.plot(x, y_feat, linewidth=1.6, label=f"smoothed (MA={smooth_win})")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Absorptance (fraction)")
ax.set_title("Absorption Spectrum")
ax.legend(loc="best")

# Mark windows
ax.axvspan(a_lo, a_hi, alpha=0.12)
ax.axvspan(b_lo, b_hi, alpha=0.12)
ax.axvspan(d_lo, d_hi, alpha=0.08)

# Mark A/B peaks
if a_pos is not None:
    ax.scatter([a_pos], [a_abs], s=55)
    ax.annotate(f"A peak\n{a_pos:.1f} nm", (a_pos, a_abs), textcoords="offset points", xytext=(10, 10))

if b_pos is not None:
    ax.scatter([b_pos], [b_abs], s=55)
    ax.annotate(f"B peak\n{b_pos:.1f} nm", (b_pos, b_abs), textcoords="offset points", xytext=(10, -20))

# Mark dip
if dip_pos is not None:
    ax.scatter([dip_pos], [dip_val], s=70)
    tag = "dip*" if dip_detected else "dip?"
    ax.annotate(f"{tag}\n{dip_pos:.1f} nm", (dip_pos, dip_val), textcoords="offset points", xytext=(10, 10))

st.pyplot(fig)

# ---------- Outputs ----------
st.subheader("厚度输出（A / B / dip）")

cA, cB, cD1, cD2 = st.columns(4)

with cA:
    st.metric("A峰厚度 (nm)", f"{tA:.2f}" if validA else "N/A")
    if a_pos is not None:
        st.caption(f"A峰：{a_pos:.1f} nm；吸收率：{a_abs:.4f}")

with cB:
    st.metric("B峰厚度 (nm)", f"{tB:.2f}" if validB else "N/A")
    if b_pos is not None:
        st.caption(f"B峰：{b_pos:.1f} nm；吸收率：{b_abs:.4f}")

with cD1:
    st.metric("dip位置厚度 (nm)", f"{tDipPos:.2f}" if np.isfinite(tDipPos) else "N/A")
    if dip_pos is not None:
        st.caption(f"dip位置：{dip_pos:.1f} nm")

with cD2:
    st.metric("dip深度厚度 (nm)", f"{tDipAbs:.2f}" if np.isfinite(tDipAbs) else "N/A")
    if dip_val is not None:
        st.caption(f"dip吸收率：{dip_val:.4f}（0–1）")

st.subheader("分段判断与建议厚度")
cZ1, cZ2, cZ3 = st.columns(3)
with cZ1:
    st.metric("分段判断", zone)
with cZ2:
    st.metric("建议厚度 (nm)", f"{suggested:.2f}" if np.isfinite(suggested) else "N/A")
with cZ3:
    st.metric("经验不确定性 (nm)", f"±{uncert:.2f}" if np.isfinite(uncert) else "N/A")

# dip diagnostics
if dip_pos is not None:
    st.caption(
        f"dip 诊断：detected={dip_detected}；"
        f"dip={dip_pos:.2f} nm, val={dip_val:.5f}"
        + (f", shoulders_avg={dip_shoulders:.5f}, drop={dip_drop_val:.5f}" if dip_drop_val is not None else "")
    )

if notes:
    st.markdown("**提示：**")
    for n in notes:
        st.write(f"- {n}")

# ---------- Data preview & export ----------
with st.expander("数据预览 / 导出（可选）", expanded=False):
    show_n = st.slider("预览行数", min_value=10, max_value=200, value=50, step=10)
    out_df = pd.DataFrame(
        {"wavelength_nm": x, "absorptance_fraction": y, "absorptance_used": y_feat}
    )
    st.dataframe(out_df.head(show_n), use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("下载处理后的数据（CSV）", data=csv_bytes, file_name="processed_spectrum.csv", mime="text/csv")

st.caption(
    "备注：本程序使用论文给出的 MoS₂ 参数（m、α）与 dip 经验式进行反演。"
    "若你的 absorptance 定义/光路/基底不同，建议用少量 AFM 真值样品进行本地标定以提升准确性。"
)

