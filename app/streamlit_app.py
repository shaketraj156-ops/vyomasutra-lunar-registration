import streamlit as st
import numpy as np
import cv2
import rasterio
import sys
import os
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'matching'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'filtering'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'warp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))
# ---- NAYA: sanity_check ka use karne ke liye path add kiya (actual folder: src/acquisition) ----
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'acquisition'))

from classical_sift import match_sift
from ransac import filter_ransac
from transform import warp_image, export_geotiff
from metrics import compute_inlier_ratio, compute_ssim
from clahe import apply_clahe
from limitations_report import generate_limitations_report
from comparison_table import generate_comparison_table
from sanity_check import validate_image_file  # ---- NAYA ----

st.set_page_config(page_title="SIH26166 — Lunar Image Registration", layout="wide")

st.title("🌙 Multi-Sensor Lunar Image Registration")
st.caption("Team VyomaSutra — SIH26166")

st.sidebar.header("Configuration")
sensor_pair = st.sidebar.selectbox(
    "Sensor Pair",
    ["OHRC – LROC NAC", "TMC-2 – LROC WAC", "IIRS – LROC WAC"]
)
apply_clahe_option = st.sidebar.checkbox("Apply CLAHE preprocessing", value=True)

col1, col2 = st.columns(2)
with col1:
    source_file = st.file_uploader("Upload Source Image (.tif)", type=["tif", "tiff"])
with col2:
    reference_file = st.file_uploader("Upload Reference Image (.tif)", type=["tif", "tiff"])

run_button = st.button("🚀 Run Pipeline", type="primary")

if run_button:
    if source_file is None or reference_file is None:
        st.error("⚠️ Pehle dono images upload karo — source aur reference.")
    else:
        # ---- NAYA: unique temp filenames — do users ek saath chalayein toh clash na ho ----
        run_id = uuid.uuid4().hex[:8]
        temp_source_path = f"temp_source_{run_id}.tif"
        temp_reference_path = f"temp_reference_{run_id}.tif"

        # ---- NAYA: poora pipeline try/finally mein — kisi bhi step pe crash ho,
        # temp files hamesha cleanup honge, aur user ko readable error milega, blank crash nahi ----
        try:
            with st.spinner("Pipeline chal raha hai..."):
                with open(temp_source_path, "wb") as f:
                    f.write(source_file.getbuffer())
                with open(temp_reference_path, "wb") as f:
                    f.write(reference_file.getbuffer())

                # ---- NAYA: upload ko pipeline mein daalne se pehle validate karo ----
                src_valid, src_info, src_issues = validate_image_file(temp_source_path)
                ref_valid, ref_info, ref_issues = validate_image_file(temp_reference_path)

                if not src_valid:
                    st.error("⚠️ Source file mein problem hai: " + "; ".join(src_issues))
                    st.stop()
                if not ref_valid:
                    st.error("⚠️ Reference file mein problem hai: " + "; ".join(ref_issues))
                    st.stop()

                try:
                    with rasterio.open(temp_source_path) as src:
                        img1 = src.read(1)
                    with rasterio.open(temp_reference_path) as src:
                        img2 = src.read(1)
                except Exception as e:
                    st.error(f"❌ Image read nahi ho payi: {e}")
                    st.stop()

                if apply_clahe_option:
                    img1_processed = apply_clahe(img1)
                    img2_processed = apply_clahe(img2)
                else:
                    img1_processed = img1
                    img2_processed = img2

                matches = match_sift(img1_processed, img2_processed)

                if len(matches) < 4:
                    st.error(f"❌ Sirf {len(matches)} matches mile — RANSAC ke liye kam se kam 4 chahiye. Behtar image pair try karo.")
                else:
                    inliers, H = filter_ransac(matches)

                    if H is None:
                        st.error("❌ Homography compute nahi ho payi — images mein enough overlap nahi hoga shayad.")
                    else:
                        rows, cols = img1_processed.shape
                        warped = warp_image(img1_processed, H, (cols, rows))

                        # ---- NAYA: warp_image ab fail hone pe None return karta hai, crash nahi ----
                        if warped is None:
                            st.error("❌ Warp step fail ho gaya — image ya homography ke saath koi issue hai.")
                            st.stop()

                        inlier_ratio = compute_inlier_ratio(len(matches), inliers)
                        ssim_score = compute_ssim(warped, img2_processed)

                        st.success("✅ Pipeline complete!")

                        st.subheader("Results")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.image(img1_processed, caption="Source (Preprocessed)", use_container_width=True)
                        with c2:
                            st.image(img2_processed, caption="Reference", use_container_width=True)
                        with c3:
                            st.image(warped, caption="Warped Source (Aligned)", use_container_width=True)

                        st.subheader("Metrics")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Matches", len(matches))
                        m2.metric("Inlier Ratio", f"{inlier_ratio:.2%}")
                        m3.metric("SSIM Score", f"{ssim_score:.4f}")

                        st.subheader("⚠️ Limitations Report")
                        try:
                            report = generate_limitations_report(
                                img1_processed, matches, inlier_ratio, ssim_score,
                                latitude=None
                            )
                            confidence_color = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}
                            st.markdown(f"**Confidence Level:** {confidence_color.get(report['confidence_level'], '')} {report['confidence_level']}")
                            for warning in report['warnings']:
                                st.warning(warning)
                        except Exception as e:
                            # ---- NAYA: limitations report fail ho toh pura pipeline result na khoye ----
                            st.warning(f"⚠️ Limitations report generate nahi ho paya: {e}")

                        st.subheader("📊 Comparison vs Makharia et al. Baseline")
                        try:
                            our_results = {
                                "test_type": "current run",
                                "combined_matches": len(matches),
                                "inliers_after_ransac": len(inliers),
                                "inlier_ratio": inlier_ratio,
                                "ssim_score": ssim_score,
                            }
                            comparison = generate_comparison_table(our_results)
                            st.caption(f"Baseline paper: {comparison['baseline_paper']}")
                            for dataset, algos in comparison['baseline_numbers'].items():
                                with st.expander(f"📁 {dataset}"):
                                    for algo, vals in algos.items():
                                        st.write(f"**{algo}**: RMSE X={vals['rmse_x']}, RMSE Y={vals['rmse_y']}, Time={vals['time_s']}s")
                            st.markdown("**Our Current Run:**")
                            oc1, oc2, oc3 = st.columns(3)
                            oc1.metric("Matches Used", our_results['combined_matches'])
                            oc2.metric("Inliers", our_results['inliers_after_ransac'])
                            oc3.metric("SSIM", f"{our_results['ssim_score']:.4f}")
                            for note in comparison['notes']:
                                st.caption(f"ℹ️ {note}")
                        except Exception as e:
                            st.warning(f"⚠️ Comparison table generate nahi ho payi: {e}")

                        # ---- NAYA: geotiff output bhi run_id se unique, aur folder create guarded ----
                        output_dir = "outputs/geotiff"
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"aligned_output_{run_id}.tif")

                        export_success = export_geotiff(
                            warped, output_path,
                            crs='EPSG:4326',
                            origin_x=45.0,
                            origin_y=-10.0,
                            pixel_size=0.001
                        )

                        if export_success:
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download Aligned Result (GeoTIFF)",
                                    data=f,
                                    file_name="aligned_result.tif",
                                    mime="image/tiff"
                                )
                        else:
                            st.warning("⚠️ GeoTIFF export failed — check console for details.")

        except Exception as e:
            # ---- NAYA: koi bhi unexpected error catch ho, user ko blank crash na dikhe ----
            st.error(f"❌ Pipeline mein unexpected error aaya: {e}")

        finally:
            # ---- NAYA: temp files hamesha cleanup ho, chahe error aaya ho ya nahi ----
            for path in (temp_source_path, temp_reference_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

else:
    st.info("👆 Dono images upload karo aur 'Run Pipeline' click karo.")