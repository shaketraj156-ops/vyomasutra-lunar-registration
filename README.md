# VyomaSutra — Multi-Sensor Lunar Image Registration

**SIH26166** — Chandrayaan-2 ke teen sensors (OHRC, TMC-2, IIRS) ki lunar images ko automatically align (register) karne wala software, alag-alag sun-angle, viewpoint, aur scale ke bawajood.

## Team

- **Shaket** — Data Acquisition, Preprocessing, Classical Matching (SIFT), Evaluation, Dashboard
- **Gaurav** — Deep Matching (LightGlue), Stretch Goals (Crater Detection)

## Problem

OHRC (0.25m), TMC-2 (5m), aur IIRS (80m) — teeno Chandrayaan-2 cameras alag-alag zoom level aur lighting conditions mein Moon ki photos lete hain. Ye software un images ko match karke ek-doosre ke upar precisely fit karta hai.

## Approach

1. **Preprocessing** — CLAHE (illumination correction), resampling, georeferencing
2. **Matching** — Classical (SIFT) + Deep learning (LightGlue) dono, side-by-side compare, combined into a unified match list
3. **Filtering** — RANSAC se galat matches hataana, uniform distribution score
4. **Refinement** — Sub-pixel accuracy
5. **Evaluation** — RMSE, inlier ratio, SSIM, comparison against Makharia et al. baseline paper
6. **Export** — Aligned output as a proper GeoTIFF (CRS + affine transform preserved)

## Current Status

🟢 **Backend pipeline ~95% complete** — full preprocessing → dual matching (SIFT + LightGlue) → RANSAC → distribution scoring → sub-pixel refinement → warp → GeoTIFF export chain is built, integrated, and tested end-to-end. Error handling and input validation added across all core modules so the pipeline degrades gracefully on corrupt/invalid input instead of crashing.

**Verified results (synthetic rotated test data):**
- 760 combined matches (22 SIFT + 738 LightGlue) on a real rotated test pair
- 99.21% inlier ratio after RANSAC filtering
- 0.9909 SSIM on the final warped/aligned output
- Distribution score 0.9116 (nearly double the SIFT-only baseline of ~0.49)

**Streamlit dashboard** is fully functional: sensor-pair selector, dual image upload, live metrics panel, limitations report, comparison table against the baseline paper, and a GeoTIFF download button.

### What's still pending

| Item | Status |
|---|---|
| Manual dual-track validation (human cross-check, with Gaurav) | Tool built and tested — joint session pending |
| Real sensor data (OHRC/TMC-2/IIRS) | Pending download by teammate |
| Comparison table with real-data numbers | Blocked on real data above |
| Public deployment (Streamlit Community Cloud) | Planned — prototype link for submission |
| Final UI polish | Planned |

## Tech Stack

Python, OpenCV, GDAL/rasterio, LightGlue, Streamlit

## Setup

```bash
conda create -n vyomasutra python=3.10 -y
conda activate vyomasutra
conda install -c conda-forge gdal rasterio -y
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Upload a source and a reference `.tif`/`.tiff` image, choose the sensor pair, and click **Run Pipeline**.

## Reference

Makharia et al., "Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data", arXiv:2509.04775 (2025)