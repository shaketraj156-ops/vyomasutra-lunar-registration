# VyomaSutra — Multi-Sensor Lunar Image Registration

**SIH26166** — Chandrayaan-2 ke teen sensors (OHRC, TMC-2, IIRS) ki lunar images ko automatically align (register) karne wala software, alag-alag sun-angle, viewpoint, aur scale ke bawajood.

## Team
- Shaket — Data Acquisition, Preprocessing, Classical Matching (SIFT), Evaluation, Dashboard
- Gaurav — Deep Matching (LightGlue), Stretch Goals (Crater Detection)

## Problem
OHRC (0.25m), TMC-2 (5m), aur IIRS (80m) — teeno Chandrayaan-2 cameras alag-alag zoom level aur lighting conditions mein Moon ki photos leते hain. Ye software un images ko match karke ek-doosre ke upar precisely fit karta hai.

## Approach
1. **Preprocessing** — CLAHE (illumination correction), resampling, georeferencing
2. **Matching** — Classical (SIFT) + Deep learning (LightGlue) dono, side-by-side compare
3. **Filtering** — RANSAC se galat matches hataana, uniform distribution score
4. **Refinement** — Sub-pixel accuracy
5. **Evaluation** — RMSE, inlier ratio, SSIM, comparison against Makharia et al. baseline paper

## Tech Stack
Python, OpenCV, GDAL/rasterio, LightGlue, Streamlit

## Setup
\`\`\`bash
conda create -n vyomasutra python=3.10 -y
conda activate vyomasutra
conda install -c conda-forge gdal rasterio -y
pip install -r requirements.txt
\`\`\`

## Status
🚧 Work in progress — Data acquisition phase

## Reference
Makharia et al., "Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data", arXiv:2509.04775 (2025)