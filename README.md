# VyomaSutra

## Multi-Sensor Lunar Image Registration

VyomaSutra is an end-to-end image registration system for aligning lunar images captured by different sensors and at different resolutions.

The project is designed for Chandrayaan-2 sensor data, including:

- OHRC: approximately 0.25 m resolution
- TMC-2: approximately 5 m resolution
- IIRS: approximately 80 m resolution
- LROC reference imagery

The system places images from different viewpoints, illumination conditions, and scales into a common reference frame. It produces an aligned, GIS-compatible GeoTIFF that can be inspected visually or used in later planetary-mapping workflows.

> **Smart India Hackathon problem:** SIH26166

## Why This Matters

Images of the Moon are captured by instruments with different resolutions, viewing angles, lighting conditions, and sensor characteristics. Directly overlaying these images can produce inaccurate results.

VyomaSutra addresses this challenge through a complete registration workflow:

1. Improve image contrast and reduce illumination differences.
2. Find corresponding features between the two images.
3. Remove incorrect feature matches.
4. Estimate the geometric transformation between the images.
5. Refine the transformation at subpixel accuracy.
6. Warp the source image onto the reference image.
7. Measure registration quality.
8. Export a georeferenced aligned product.

## Key Features

### Multi-sensor registration

The dashboard provides profiles for OHRC-LROC, TMC-2-LROC, and IIRS-LROC image pairs. Sensor-specific preprocessing values are stored in YAML configuration.

### Illumination-aware preprocessing

CLAHE, or Contrast Limited Adaptive Histogram Equalization, improves local contrast and helps reduce the effect of lunar shadows and changing illumination.

### Classical and deep matching

- **Classical baseline:** OpenCV SIFT with Lowe's ratio test.
- **Deep matching:** SuperPoint features with LightGlue matching through PyTorch.
- **Dual matching:** Combines classical and deep matches when LightGlue is available.

### Robust geometric estimation

RANSAC rejects geometrically inconsistent matches and estimates a homography. This prevents a small number of incorrect matches from dominating the alignment.

### Subpixel refinement

OpenCV `cornerSubPix` refines feature coordinates in both images. The homography is then estimated again using the refined coordinates.

### GIS-ready output

The aligned result is exported as a single-band GeoTIFF. When available, the reference image's CRS and affine transform are preserved in the output.

### Scientific evaluation

The dashboard reports:

- Total matches
- RANSAC inliers
- Inlier ratio
- Reprojection RMSE
- Structural similarity, or SSIM
- Spatial match distribution score

### Visual inspection tools

The Streamlit dashboard includes:

- Source, reference, and warped-image comparison
- Dynamic checkerboard overlay
- Feature correspondence vectors
- Baseline comparison table
- Terrain and registration limitations report
- Downloadable aligned GeoTIFF

## System Workflow

```text
Source TIFF + Reference TIFF
              |
              v
       Rasterio image loading
              |
              v
     CLAHE and bit-depth normalization
              |
              v
       SIFT / LightGlue matching
              |
              v
       RANSAC outlier rejection
              |
              v
       Subpixel point refinement
              |
              v
       Homography re-estimation
              |
              v
       OpenCV perspective warping
              |
              v
       Metrics and risk analysis
              |
              v
     Aligned GeoTIFF and dashboard output
```

## Project Structure

```text
app/
    streamlit_app.py          Interactive dashboard and user workflow

src/
    pipeline.py               Main registration pipeline
    config/sensors.yaml       Active sensor configuration

    preprocessing/            CLAHE, georeferencing, and resampling
    matching/                 SIFT, LightGlue, and dual matching
    filtering/                RANSAC and match distribution scoring
    refinement/               Subpixel keypoint refinement
    warp/                     Warping, checkerboards, and GeoTIFF export
    evaluation/               Metrics, validation, and limitations reports
    acquisition/              Synthetic data and raster sanity checks
```

## Technology Stack

- Python 3.10
- Streamlit for the interactive dashboard
- NumPy for numerical processing
- OpenCV and OpenCV Contrib for SIFT, CLAHE, RANSAC, refinement, and warping
- Rasterio and GDAL for GeoTIFF and geospatial metadata
- PyTorch and Torchvision for deep feature matching
- LightGlue and SuperPoint for learned feature matching
- scikit-image for SSIM evaluation
- PyYAML for sensor configuration

## Installation

Conda is recommended on Windows because GDAL and Rasterio use native geospatial libraries.

```powershell
conda create -n vyomasutra python=3.10 -y
conda activate vyomasutra
conda install -c conda-forge gdal rasterio -y
pip install -r requirements.txt
```

Additional requirements:

- Git is needed because LightGlue is installed from GitHub.
- Internet access may be needed for dependency installation and model weights.
- A GPU is optional. PyTorch can run on CPU, although deep matching will be slower.
- Input files should be readable single-band `.tif` or `.tiff` images.
- Georeferenced inputs with CRS and affine-transform metadata are recommended.

## Run the Dashboard

From the repository root:

```powershell
conda activate vyomasutra
streamlit run app/streamlit_app.py
```

Open the local address shown by Streamlit, normally:

```text
http://localhost:8501
```

### Using custom images

1. Select **Upload Custom TIFF Files**.
2. Upload the moving/source TIFF.
3. Upload the fixed/reference TIFF.
4. Select the sensor pair.
5. Select SIFT, LightGlue, or the dual matcher.
6. Configure CLAHE, subpixel refinement, and the RANSAC threshold if needed.
7. Click **Run Registration Pipeline**.
8. Inspect the metrics and visual tabs.
9. Download the aligned GeoTIFF.

## Generate Demo Data

The repository includes a synthetic-data generator for local demonstrations. It creates a crater-rich lunar terrain pair with a known rotation, translation, contrast change, CRS, and affine transform.

```powershell
conda activate vyomasutra
python src/acquisition/create_sample_data.py
streamlit run app/streamlit_app.py
```

The generated files are:

```text
data/samples/lunar_source_crater.tif
data/samples/lunar_reference_crater.tif
```

These files are useful for testing the workflow, but they are not a replacement for real OHRC, TMC-2, IIRS, or LROC data.


## Current Implementation Status

### Working

- Streamlit dashboard
- TIFF upload workflow
- Synthetic TIFF generation
- CLAHE preprocessing
- SIFT matching
- RANSAC homography estimation
- Subpixel refinement
- Perspective warping
- Registration metrics
- Checkerboard and match-vector visualization
- GeoTIFF export with reference spatial metadata
- Limitations and confidence reporting



## Important Technical Notes


### Synthetic versus real evaluation

Synthetic data provides known transformations and is useful for validating pipeline logic. Final scientific performance should be measured on real lunar sensor data with different resolutions, sun angles, viewpoints, and terrain types.

### Geospatial metadata

The reference image is treated as the fixed spatial frame. The source image is warped to the reference pixel canvas, and the reference CRS and affine transform are inherited when available.

## Research Reference

The dashboard includes a comparison against reported results from:

> Makharia et al., "Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data," arXiv:2509.04775, 2025.

The comparison is informative rather than a direct benchmark until the same real scenes are processed by both systems.

## Team

- **Shaket:** Data acquisition, preprocessing, classical matching, evaluation, and dashboard
- **Gaurav:** Deep matching with LightGlue and future crater-detection extensions
- **Shubh Sahu:** Project integration, local validation, technical analysis, and documentation
