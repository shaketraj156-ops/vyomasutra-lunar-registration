"""
streamlit_app.py
VyomaSutra — Multi-Sensor Lunar Image Registration (SIH 2026 / PS 26166)
Interactive, Judge-Ready Dashboard with Dual Matching, Sub-pixel Refinement,
Checkerboard Alignment, and Authentic GIS GeoTIFF Export.
"""

import streamlit as st
import numpy as np
import cv2
import rasterio
import sys
import os
import uuid
import json
import time

# Ensure internal modules are discoverable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
for folder in ["matching", "filtering", "refinement", "warp", "evaluation", "preprocessing", "acquisition"]:
    p = os.path.join(SRC_DIR, folder)
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import run_registration_pipeline
from transform import create_checkerboard_overlay, export_geotiff
from comparison_table import generate_comparison_table

# Page Configuration
st.set_page_config(
    page_title="VyomaSutra | Lunar Image Registration",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Space-Grade Dark UI Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-subpixel {
        background-color: #065F46;
        color: #34D399;
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🌙 VyomaSutra — Lunar Image Registration Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Chandrayaan-2 (OHRC / TMC-2 / IIRS) Multi-Modal, Sun-Angle & Scale-Invariant Alignment | SIH 2026 (PS 26166)</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=120)
st.sidebar.markdown("### ⚙️ Pipeline Configuration")

input_mode = st.sidebar.radio(
    "Input Mode",
    ["🎯 Preloaded Lunar Demo Pair (Instant Demo)", "📂 Upload Custom TIFF Files"],
    index=0
)

sensor_pair = st.sidebar.selectbox(
    "Sensor Pair Profile",
    [
        "OHRC — LROC NAC (High-Resolution Zoom: 0.25m)",
        "TMC-2 — LROC WAC (Terrain Mapping: 5m)",
        "IIRS — LROC WAC (Infrared Hyperspectral: 80m)"
    ],
    index=0
)

matcher_choice = st.sidebar.selectbox(
    "Matching Engine",
    [
        "Dual Ensemble (SuperPoint+LightGlue + SIFT)",
        "Deep LightGlue + SuperPoint",
        "Classical SIFT Baseline"
    ],
    index=0
)

# Convert UI selection to backend keys
sensor_key_map = {
    "OHRC": "OHRC_LROC",
    "TMC-2": "TMC_LROC",
    "IIRS": "IIRS_LROC"
}
selected_sensor_prefix = sensor_pair.split(" ")[0]
backend_sensor_key = sensor_key_map.get(selected_sensor_prefix, "OHRC_LROC")

backend_matcher_map = {
    "Dual Ensemble (SuperPoint+LightGlue + SIFT)": "dual",
    "Deep LightGlue + SuperPoint": "lightglue",
    "Classical SIFT Baseline": "sift"
}
backend_matcher_type = backend_matcher_map.get(matcher_choice, "dual")

with st.sidebar.expander("🛠️ Advanced Parameters", expanded=False):
    apply_clahe = st.checkbox("CLAHE Illumination Equalization", value=True, help="Equalizes extreme lunar shadow-contrast")
    enable_subpixel = st.checkbox("Sub-pixel Keypoint Refinement", value=True, help="Refines detected points to sub-pixel coordinates")
    ransac_thresh = st.slider("RANSAC Inlier Threshold (px)", 1.0, 10.0, 3.0, 0.5)

# Load / Upload Data
source_array = None
reference_array = None
ref_crs = None
ref_transform = None
source_name = "Source Image"
reference_name = "Reference Image"

sample_dir = os.path.join(ROOT_DIR, "data", "samples")
src_sample_path = os.path.join(sample_dir, "lunar_source_crater.tif")
ref_sample_path = os.path.join(sample_dir, "lunar_reference_crater.tif")

if input_mode == "🎯 Preloaded Lunar Demo Pair (Instant Demo)":
    if os.path.exists(src_sample_path) and os.path.exists(ref_sample_path):
        with rasterio.open(src_sample_path) as s:
            source_array = s.read(1)
        with rasterio.open(ref_sample_path) as r:
            reference_array = r.read(1)
            ref_crs = r.crs
            ref_transform = r.transform
        source_name = "Lunar Crater Field (Rotated & Contrast Variant)"
        reference_name = "Lunar Reference Basin (IAU2000 GIS Frame)"
        st.info("💡 **Demo Pair Loaded:** High-contrast crater terrain with synthetic sun-angle illumination variation and 5° orbital rotation.")
    else:
        st.warning("Sample pair not found on disk. Please upload custom files.")
else:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        s_file = st.file_uploader("Upload Moving / Source (.tif)", type=["tif", "tiff"])
    with col_up2:
        r_file = st.file_uploader("Upload Fixed / Reference (.tif)", type=["tif", "tiff"])

    if s_file and r_file:
        run_id = uuid.uuid4().hex[:6]
        tmp_s = f"temp_s_{run_id}.tif"
        tmp_r = f"temp_r_{run_id}.tif"
        with open(tmp_s, "wb") as f:
            f.write(s_file.getbuffer())
        with open(tmp_r, "wb") as f:
            f.write(r_file.getbuffer())

        try:
            with rasterio.open(tmp_s) as s:
                source_array = s.read(1)
            with rasterio.open(tmp_r) as r:
                reference_array = r.read(1)
                ref_crs = r.crs
                ref_transform = r.transform
            source_name = s_file.name
            reference_name = r_file.name
        finally:
            for p in (tmp_s, tmp_r):
                if os.path.exists(p):
                    os.remove(p)

# Execution Button
run_pipeline_btn = st.button("🚀 Run Registration Pipeline", type="primary", use_container_width=True)

if run_pipeline_btn:
    if source_array is None or reference_array is None:
        st.error("⚠️ Please select or upload both source and reference images.")
    else:
        with st.spinner(f"Aligning lunar imagery using {matcher_choice}..."):
            t_start = time.time()
            result = run_registration_pipeline(
                source_img=source_array,
                reference_img=reference_array,
                sensor_pair_key=backend_sensor_key,
                matcher_type=backend_matcher_type,
                apply_clahe_flag=apply_clahe,
                ransac_threshold=ransac_thresh,
                enable_subpixel=enable_subpixel,
                ref_crs=ref_crs,
                ref_transform=ref_transform
            )
            t_elapsed = time.time() - t_start

        if not result["success"]:
            st.error(f"❌ Alignment Failed: {result.get('error_message')}")
        else:
            metrics = result["metrics"]
            st.success(f"✅ **Registration Succeeded** in {result['elapsed_time_s']}s!")

            # 1. Scientific KPI Dashboard
            st.markdown("### 📊 Registration Metrics Dashboard")
            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            
            with m_col1:
                st.metric("Total Matches", f"{metrics['total_matches']:,}")
            with m_col2:
                st.metric("RANSAC Inliers", f"{metrics['inlier_count']:,}")
            with m_col3:
                st.metric("Inlier Ratio", f"{metrics['inlier_ratio']:.1%}")
            with m_col4:
                rmse_val = metrics['reprojection_rmse']
                st.metric("Reprojection RMSE", f"{rmse_val:.3f} px")
            with m_col5:
                st.metric("Structural SSIM", f"{metrics['ssim_score']:.4f}")
            with m_col6:
                st.metric("Distribution Score", f"{metrics['distribution_score']:.3f}")

            # 2. Tabbed Visual Inspection
            tab_checker, tab_align, tab_vectors, tab_benchmark, tab_limitations = st.tabs([
                "🏁 Checkerboard Alignment",
                "🖼️ Side-by-Side Comparison",
                "📍 Match Vectors",
                "📈 Baseline Benchmark (Makharia et al.)",
                "⚠️ Limitations & Terrain Report"
            ])

            # Tab 1: Checkerboard
            with tab_checker:
                st.markdown("#### Dynamic Checkerboard Mosaic")
                st.caption("Crater rims and geological boundaries should line up seamlessly across alternating squares.")
                tile_size = st.slider("Checkerboard Tile Size (pixels)", 16, 128, 48, 8)
                chk_img = create_checkerboard_overlay(result["warped_image"], result["reference_processed"], tile_size=tile_size)
                st.image(chk_img, caption=f"Checkerboard Blend ({tile_size}px grid) — Inspect edge continuity", use_container_width=True)

            # Tab 2: Side-by-Side
            with tab_align:
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.image(result["source_processed"], caption="Source / Moving (Preprocessed)", use_container_width=True)
                with col_v2:
                    st.image(result["reference_processed"], caption="Reference / Fixed (Preprocessed)", use_container_width=True)
                with col_v3:
                    st.image(result["warped_image"], caption="Warped Source (Aligned to Reference)", use_container_width=True)

            # Tab 3: Match Vectors
            with tab_vectors:
                st.markdown("#### Feature Correspondence Vectors")
                matches_to_plot = result["inliers"][:120]  # Plot top 120 for visual clarity
                h_max = max(result["source_processed"].shape[0], result["reference_processed"].shape[0])
                w1 = result["source_processed"].shape[1]
                w2 = result["reference_processed"].shape[1]
                canvas = np.zeros((h_max, w1 + w2), dtype=np.uint8)
                canvas[:result["source_processed"].shape[0], :w1] = result["source_processed"]
                canvas[:result["reference_processed"].shape[0], w1:w1+w2] = result["reference_processed"]
                canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

                for m in matches_to_plot:
                    pt1 = (int(round(m[0])), int(round(m[1])))
                    pt2 = (int(round(m[2])) + w1, int(round(m[3])))
                    # Color by engine tag if available
                    color = (0, 255, 128) if len(m) > 5 and m[5] == "lightglue" else (0, 200, 255)
                    cv2.circle(canvas_rgb, pt1, 3, color, -1)
                    cv2.circle(canvas_rgb, pt2, 3, color, -1)
                    cv2.line(canvas_rgb, pt1, pt2, color, 1, cv2.LINE_AA)

                st.image(canvas_rgb, caption=f"Inlier Correspondences (Showing {len(matches_to_plot)} of {len(result['inliers'])} points)", use_container_width=True)

            # Tab 4: Baseline Benchmark
            with tab_benchmark:
                st.markdown("#### Comparative Benchmark vs ISRO Space Applications Centre (SAC) Baseline")
                our_eval = {
                    "test_type": f"SIH 2026 Current Run ({matcher_choice})",
                    "total_matches": metrics["total_matches"],
                    "inliers_after_ransac": metrics["inlier_count"],
                    "inlier_ratio": metrics["inlier_ratio"],
                    "reprojection_rmse": metrics["reprojection_rmse"],
                    "distribution_score": metrics["distribution_score"],
                    "ssim_score": metrics["ssim_score"]
                }
                bench_data = generate_comparison_table(our_eval)
                st.caption(f"Reference: {bench_data['baseline_paper']}")
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown("**Published SAC Paper Results (Makharia et al., arXiv:2509.04775):**")
                    for scene_id, algs in bench_data["baseline_numbers"].items():
                        with st.expander(f"📌 {scene_id}"):
                            for algo, vals in algs.items():
                                st.write(f"- **{algo}**: RMSE X={vals['rmse_x']}, RMSE Y={vals['rmse_y']}, Compute Time={vals['time_s']}s")
                
                with col_b2:
                    st.markdown("**VyomaSutra Enhanced Solution:**")
                    st.write(f"- **Matching Method**: {matcher_choice}")
                    st.write(f"- **Inliers Recovered**: {metrics['inlier_count']} / {metrics['total_matches']}")
                    st.write(f"- **Reprojection RMSE**: `{metrics['reprojection_rmse']:.4f} px`")
                    st.write(f"- **Distribution Score**: `{metrics['distribution_score']:.4f}`")
                    st.write(f"- **Structural SSIM**: `{metrics['ssim_score']:.4f}`")
                    for n in bench_data["notes"]:
                        st.caption(f"ℹ️ {n}")

            # Tab 5: Limitations & Planetary Report
            with tab_limitations:
                st.markdown("#### Autonomous Planetary Risk Assessment")
                lims = result.get("limitations", {})
                conf_level = lims.get("confidence_level", "HIGH")
                conf_emoji = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(conf_level, "⚪")
                st.markdown(f"**Confidence Classification:** {conf_emoji} **{conf_level}**")
                
                warnings = lims.get("warnings", [])
                if warnings:
                    for w in warnings:
                        st.warning(w)
                else:
                    st.success("No significant terrain risks detected. Good feature density and illumination contrast.")

            # 3. GeoTIFF Export Section
            st.markdown("---")
            st.markdown("### 💾 GIS Product Delivery")
            out_dir = os.path.join(ROOT_DIR, "outputs", "geotiff")
            os.makedirs(out_dir, exist_ok=True)
            export_path = os.path.join(out_dir, f"vyomasutra_aligned_{uuid.uuid4().hex[:6]}.tif")

            exported = export_geotiff(
                result["warped_image"],
                export_path,
                ref_crs=ref_crs,
                ref_transform=ref_transform
            )

            if exported and os.path.exists(export_path):
                with open(export_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Aligned Lunar GeoTIFF (GIS Compatible)",
                        data=f,
                        file_name="vyomasutra_lunar_aligned.tif",
                        mime="image/tiff",
                        type="primary"
                    )
                st.caption("✅ GeoTIFF includes true spatial affine transformation and Lunar cartographic CRS inherited from the reference dataset.")