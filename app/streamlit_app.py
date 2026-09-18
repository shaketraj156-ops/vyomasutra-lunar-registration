"""
app.py — VyomaSutra: Lunar Image Registration Engine
Streamlit application featuring an interactive, full-screen 3D revolving Moon
background built with Three.js, floating glassmorphic UI, Dual Matching Engine,
Sub-pixel Refinement, Multi-Metric Gatekeeper, and GIS GeoTIFF Export.
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
import streamlit.components.v1 as components

# Ensure internal modules are discoverable
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if not os.path.exists(SRC_DIR):
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
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

try:
    from assets.moon_texture_data import NASA_MOON_MAP_B64, NASA_MOON_BUMP_B64
except Exception:
    NASA_MOON_MAP_B64 = ""
    NASA_MOON_BUMP_B64 = ""

try:
    from assets.video_texture_data import LOCAL_VIDEO_B64
except Exception:
    LOCAL_VIDEO_B64 = ""

try:
    from assets.team_logo_b64 import TEAM_LOGO_B64
except Exception:
    TEAM_LOGO_B64 = ""

# Page Configuration
st.set_page_config(
    page_title="VyomaSutra | Lunar Image Registration Engine",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_threejs_moon_background():
    """
    Injects custom CSS to make Streamlit container backgrounds transparent
    and runs a Three.js WebGL script that renders a realistic 3D revolving Moon
    centered in deep space with 3,000 twinkling background stars.
    """
    # 1. Glassmorphism & Transparency CSS
    st.markdown("""
    <style>
    /* Force full transparency across all Streamlit containers */
    html, body, .stApp, [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], header, [data-testid="stMain"], 
    .main, section.main, .stMainBlockContainer {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    body {
        background-color: #03050a !important;
    }

    /* Completely hide Streamlit sidebar and toggle button */
    [data-testid="stSidebar"], 
    section[data-testid="stSidebar"], 
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    div[data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        max-width: 0 !important;
        min-width: 0 !important;
    }

    [data-testid="stMain"], 
    .stMainBlockContainer, 
    div[data-testid="stAppViewBlockContainer"], 
    section.main {
        margin-left: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        padding-top: 0.5rem !important;
    }
    
    /* Ensure main content floats above the 3D Moon canvas */
    [data-testid="stMain"] {
        position: relative !important;
        z-index: 10 !important;
    }

    /* HD High-Contrast Typography & Text Clarity Overrides */
    p, span, label, .stMarkdown, div[data-testid="stText"], [data-testid="stMarkdownContainer"] {
        color: #F8FAFC !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.95) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.95), 0 0 15px rgba(0, 163, 255, 0.35) !important;
    }

    /* Main Header Title HD Container */
    .main-title {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(11, 16, 29, 0.88) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 12px !important;
        padding: 14px 22px !important;
        margin-bottom: 20px !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7), 0 0 20px rgba(56, 189, 248, 0.25) !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.95) !important;
    }

    /* Solid Pitch-Dark Card Container for Demo Pair Loaded Banner (st.info) */
    .stAlert, 
    [data-testid="stAlert"], 
    [data-testid="stNotification"], 
    div[data-baseweb="notification"], 
    [data-testid="stAlertContainer"] {
        background-color: #03060d !important;
        background: #03060d !important;
        border: 2px solid #00A3FF !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.98), 0 0 22px rgba(0, 163, 255, 0.55) !important;
        padding: 14px 22px !important;
        opacity: 1.0 !important;
    }

    /* Force PURE BRIGHT WHITE Text inside Demo Pair Banner & Alerts */
    .stAlert *, 
    [data-testid="stAlert"] *, 
    [data-testid="stNotification"] *, 
    div[data-baseweb="notification"] *, 
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 1.0) !important;
        opacity: 1.0 !important;
    }

    /* HD Alert Text Highlights & Bold Labels */
    .stAlert strong, 
    [data-testid="stAlert"] strong,
    [data-testid="stNotification"] strong, 
    .stAlert b,
    [data-testid="stAlert"] b {
        color: #38BDF8 !important;
        -webkit-text-fill-color: #38BDF8 !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.85) !important;
    }

    /* Hide Streamlit default status widgets & footers safely */
    div[data-testid="stStatusWidget"], footer, #MainMenu {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Glassmorphism Upload Cards */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px dashed #00A3FF !important;
        border-radius: 14px !important;
        backdrop-filter: blur(14px);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 25px rgba(0, 163, 255, 0.4);
    }
    
    /* Target ALL Streamlit Buttons including Primary & Secondary buttons */
    .stButton button, 
    button[data-testid="stBaseButton-primary"], 
    button[data-testid="stBaseButton-secondary"],
    button[kind="primary"],
    button[kind="secondary"],
    div.stButton > button, 
    div[data-testid="stButton"] > button {
        width: 100% !important;
        background: linear-gradient(90deg, #0284C7 0%, #00A3FF 50%, #0284C7 100%) !important;
        background-color: #00A3FF !important;
        border: 1.5px solid #FFFFFF !important;
        border-radius: 10px !important;
        padding: 14px 28px !important;
        color: #FFFFFF !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.8px !important;
        text-align: center !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        box-shadow: 0 6px 25px rgba(0, 163, 255, 0.75), 0 0 15px rgba(56, 189, 248, 0.5) !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.9) !important;
        transition: all 0.3s ease-in-out !important;
        cursor: pointer !important;
    }

    .stButton button:hover, 
    button[data-testid="stBaseButton-primary"]:hover, 
    button[data-testid="stBaseButton-secondary"]:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    div.stButton > button:hover {
        background: linear-gradient(90deg, #00A3FF 0%, #38BDF8 50%, #00A3FF 100%) !important;
        background-color: #38BDF8 !important;
        border-color: #FFFFFF !important;
        color: #FFFFFF !important;
        box-shadow: 0 10px 35px rgba(0, 163, 255, 0.95), 0 0 25px rgba(56, 189, 248, 0.85) !important;
        text-shadow: 0 0 12px rgba(255, 255, 255, 0.9) !important;
        transform: translateY(-2px) scale(1.01) !important;
    }

    .stButton button:focus,
    button[data-testid="stBaseButton-primary"]:focus,
    button[data-testid="stBaseButton-secondary"]:focus,
    button[kind="primary"]:focus,
    div.stButton > button:focus,
    div.stButton > button:active,
    div.stButton > button:focus-visible {
        outline: none !important;
        border-color: #FFFFFF !important;
        box-shadow: 0 0 25px rgba(0, 163, 255, 0.9) !important;
    }

    /* Sidebar HD Styling */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.9) !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #38BDF8 !important;
        font-weight: 800 !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.5) !important;
    }

    /* Glass Panels for Metrics */
    div[data-testid="stMetric"] {
        background-color: rgba(15, 23, 42, 0.88) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 12px;
        backdrop-filter: blur(14px);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.4) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #CBD5E1;
        font-weight: 700;
        background-color: rgba(15, 23, 42, 0.75);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Realistic Centered 3D Moon + 3,000 Starfield Injection via components.html
    js_code = """
    <script>
    (function() {
        var pWin = window.parent || window;
        var pDoc = pWin.document;

        // Force transparent backgrounds in parent document
        var styleId = 'vyoma-3d-moon-css';
        if (!pDoc.getElementById(styleId)) {
            var style = pDoc.createElement('style');
            style.id = styleId;
            style.innerHTML = `
                html, body, .stApp, [data-testid="stAppViewContainer"], 
                [data-testid="stHeader"], header, [data-testid="stMain"], 
                .main, section.main, .stMainBlockContainer {
                    background: transparent !important;
                    background-color: transparent !important;
                }
                body {
                    background-color: #03050a !important;
                }
                #moon-background-canvas {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    z-index: 2 !important;
                    pointer-events: none !important;
                }
                #vyoma-yt-bg-video {
                    position: fixed !important;
                    top: 50% !important;
                    left: 50% !important;
                    width: 177.77777778vh !important;
                    min-width: 100vw !important;
                    height: 56.25vw !important;
                    min-height: 100vh !important;
                    transform: translate(-50%, -50%) !important;
                    z-index: 1 !important;
                    pointer-events: none !important;
                    border: none !important;
                    filter: brightness(0.85) contrast(1.05) !important;
                }
                [data-testid="stMain"], [data-testid="stSidebar"] {
                    position: relative !important;
                    z-index: 10 !important;
                }
            `;
            pDoc.head.appendChild(style);
        }

        // Hide YouTube background iframe if present
        var ytIframe = pDoc.getElementById('vyoma-yt-bg-video');
        if (ytIframe) ytIframe.style.display = 'none';

        var canvas = pDoc.getElementById('moon-background-canvas');
        if (!canvas) {
            canvas = pDoc.createElement('canvas');
            canvas.id = 'moon-background-canvas';
            pDoc.body.appendChild(canvas);
        } else {
            canvas.style.display = 'block';
        }

        if (pWin._vyomaMoonActive) return;
        pWin._vyomaMoonActive = true;

        function startThree() {
            if (!pWin.THREE) return;
            var THREE = pWin.THREE;

            var scene = new THREE.Scene();
            var camera = new THREE.PerspectiveCamera(50, pWin.innerWidth / pWin.innerHeight, 0.1, 1000);
            
            var renderer = new THREE.WebGLRenderer({ 
                canvas: canvas, 
                alpha: true, 
                antialias: true 
            });
            renderer.setPixelRatio(Math.min(pWin.devicePixelRatio, 2));
            renderer.setSize(pWin.innerWidth, pWin.innerHeight);
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.15;

            // 1. STARFIELD PARTICLES (4,500 Twinkling Stars surrounding the Moon & overlaying video)
            var starsCount = 4500;
            var starGeometry = new THREE.BufferGeometry();
            var starPositions = new Float32Array(starsCount * 3);
            var starColors = new Float32Array(starsCount * 3);

            for (var i = 0; i < starsCount; i++) {
                var i3 = i * 3;
                var r = 90 + Math.random() * 510;
                var theta = Math.random() * Math.PI * 2;
                var phi = Math.acos((Math.random() * 2) - 1);
                
                starPositions[i3]     = r * Math.sin(phi) * Math.cos(theta);
                starPositions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
                starPositions[i3 + 2] = r * Math.cos(phi);

                var colChoice = Math.random();
                if (colChoice > 0.85) {
                    starColors[i3] = 0.60; starColors[i3+1] = 0.85; starColors[i3+2] = 1.0;
                } else if (colChoice > 0.65) {
                    starColors[i3] = 1.0;  starColors[i3+1] = 0.92; starColors[i3+2] = 0.75;
                } else {
                    starColors[i3] = 0.95; starColors[i3+1] = 0.97; starColors[i3+2] = 1.0;
                }
            }

            starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
            starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

            var starMaterial = new THREE.PointsMaterial({
                size: 1.35,
                vertexColors: true,
                transparent: true,
                opacity: 0.88,
                sizeAttenuation: true
            });

            var starField = new THREE.Points(starGeometry, starMaterial);
            scene.add(starField);

            // 2. LARGE REALISTIC CENTERED 3D MOON (MATCHING YOUTUBE VIDEO FRAMING & NASA LROC MAPS)
            var geometry = new THREE.SphereGeometry(15.5, 128, 128);
            
            var mapDataB64 = "__NASA_MAP_B64__";
            var bumpDataB64 = "__NASA_BUMP_B64__";

            var textureLoader = new THREE.TextureLoader();
            var moonTexture = null;
            var moonBumpMap = null;

            if (mapDataB64 && mapDataB64.length > 100) {
                moonTexture = textureLoader.load('data:image/jpeg;base64,' + mapDataB64);
                moonTexture.wrapS = THREE.RepeatWrapping;
                moonTexture.wrapT = THREE.ClampToEdgeWrapping;
            }

            if (bumpDataB64 && bumpDataB64.length > 100) {
                moonBumpMap = textureLoader.load('data:image/jpeg;base64,' + bumpDataB64);
                moonBumpMap.wrapS = THREE.RepeatWrapping;
                moonBumpMap.wrapT = THREE.ClampToEdgeWrapping;
            }

            var material = new THREE.MeshStandardMaterial({ 
                map: moonTexture,
                bumpMap: moonBumpMap,
                bumpScale: 0.16,
                roughness: 0.94,
                metalness: 0.02
            });

            // Online High-Res Texture fallback override
            textureLoader.load(
                'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r128/examples/textures/planets/moon_1024.jpg',
                function(onlineTex) {
                    onlineTex.wrapS = THREE.RepeatWrapping;
                    onlineTex.wrapT = THREE.ClampToEdgeWrapping;
                    material.map = onlineTex;
                    material.needsUpdate = true;
                }
            );

            var moon = new THREE.Mesh(geometry, material);
            // Centered Large Moon matching YouTube video frame exactly
            moon.position.set(0, 0, -5);
            scene.add(moon);

            // 3. ATMOSPHERIC LIMB GLOW HALO
            var haloGeometry = new THREE.SphereGeometry(15.9, 64, 64);
            var haloMaterial = new THREE.MeshBasicMaterial({
                color: 0x38bdf8,
                side: THREE.BackSide,
                transparent: true,
                opacity: 0.10
            });
            var halo = new THREE.Mesh(haloGeometry, haloMaterial);
            halo.position.set(0, 0, -5);
            scene.add(halo);

            // 4. SPACE LIGHTING (Matching Video Directional Sunlight from top-right)
            var sunLight = new THREE.DirectionalLight(0xfff8ee, 2.4);
            sunLight.position.set(45, 15, 30);
            scene.add(sunLight);

            var ambientLight = new THREE.AmbientLight(0x142032, 0.40);
            scene.add(ambientLight);

            camera.position.z = 26;

            // 5. ANIMATION LOOP: Continuous Moon Revolution & Star Twinkle Rotation
            function animate() {
                pWin.requestAnimationFrame(animate);
                
                // Y-axis revolution of the centered Moon matching reference video
                moon.rotation.y += 0.0018;
                moon.rotation.x += 0.0001;
                
                // Rotation of starfield background
                starField.rotation.y += 0.0002;
                starField.rotation.x += 0.0001;
                
                renderer.render(scene, camera);
            }
            animate();

            pWin.addEventListener('resize', function() {
                var w = pWin.innerWidth;
                var h = pWin.innerHeight;
                renderer.setSize(w, h);
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
            });
        }

        // Overlay Team Logo badge directly in bottom-right corner
        var badgeId = 'vyoma-lunar-flux-bottom-badge';
        var badge = pDoc.getElementById(badgeId);
        var logoB64 = "__TEAM_LOGO_B64__";

        if (!badge && logoB64.length > 50) {
            badge = pDoc.createElement('div');
            badge.id = badgeId;
            badge.style.cssText = `
                position: fixed !important;
                bottom: 30px !important;
                right: 40px !important;
                z-index: 999999 !important;
                width: 85px !important;
                height: 85px !important;
                padding: 8px !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                background-color: #03060d !important;
                background: radial-gradient(circle, rgba(11, 16, 29, 0.98) 0%, rgba(3, 6, 13, 0.98) 100%) !important;
                border: 2px solid #00A3FF !important;
                border-radius: 14px !important;
                box-shadow: 0 0 25px rgba(0, 163, 255, 0.85), 0 8px 30px rgba(0, 0, 0, 0.95) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                pointer-events: auto !important;
            `;
            badge.innerHTML = `
                <img src="data:image/png;base64,` + logoB64 + `" style="width: 100%; height: 100%; object-fit: contain; filter: drop-shadow(0 0 8px rgba(0, 163, 255, 0.9));" alt="Lunar Flux Logo">
            `;
            pDoc.body.appendChild(badge);
        } else if (badge) {
            badge.style.display = 'flex';
        }

        if (!pWin.THREE) {
            var script = pDoc.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
            script.onload = startThree;
            pDoc.head.appendChild(script);
        } else {
            startThree();
        }
    })();
    </script>
    """
    js_code = js_code.replace("__NASA_MAP_B64__", NASA_MOON_MAP_B64).replace("__NASA_BUMP_B64__", NASA_MOON_BUMP_B64).replace("__TEAM_LOGO_B64__", TEAM_LOGO_B64)
    components.html(js_code, height=0, width=0)

def inject_local_mp4_background_video():
    """
    Safely injects background video and floating Team LUNAR FLUX logo badge 
    behind the glassmorphic Streamlit UI without breaking DOM rendering.
    """
    js_code = """
    <script>
    (function() {
        var pWin = window.parent || window;
        var pDoc = pWin.document;

        var styleId = 'vyoma-safe-bg-css';
        if (!pDoc.getElementById(styleId)) {
            var style = pDoc.createElement('style');
            style.id = styleId;
            style.innerHTML = `
                html, body, .stApp {
                    background-color: #020408 !important;
                }
                .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                    background: transparent !important;
                    background-color: transparent !important;
                }
                #vyoma-bg-video {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    object-fit: cover !important;
                    z-index: -1 !important;
                    pointer-events: none !important;
                    filter: brightness(0.70) contrast(1.10) !important;
                }
                #vyoma-lunar-flux-badge {
                    position: fixed !important;
                    bottom: 30px !important;
                    right: 35px !important;
                    z-index: 999999 !important;
                    width: 80px !important;
                    height: 80px !important;
                    padding: 6px !important;
                    box-sizing: border-box !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    background: radial-gradient(circle, rgba(11, 16, 29, 0.98) 0%, rgba(3, 6, 13, 0.98) 100%) !important;
                    border: 2px solid #00A3FF !important;
                    border-radius: 14px !important;
                    box-shadow: 0 0 25px rgba(0, 163, 255, 0.85), 0 8px 30px rgba(0, 0, 0, 0.95) !important;
                    backdrop-filter: blur(16px) !important;
                    -webkit-backdrop-filter: blur(16px) !important;
                }
            `;
            pDoc.head.appendChild(style);
        }

        // 1. Video Background Injection
        var video = pDoc.getElementById('vyoma-bg-video');
        var videoB64 = "__LOCAL_VIDEO_B64__";

        if (!video && videoB64 && videoB64.length > 100) {
            video = pDoc.createElement('video');
            video.id = 'vyoma-bg-video';
            video.autoplay = true;
            video.loop = true;
            video.muted = true;
            video.defaultMuted = true;
            video.playsInline = true;
            video.setAttribute('playsinline', '');
            video.setAttribute('muted', '');
            video.setAttribute('autoplay', '');
            video.setAttribute('loop', '');
            video.src = "data:video/mp4;base64," + videoB64;
            pDoc.body.prepend(video);
        }

        if (video) {
            video.play().catch(function(e) {});
        }

        // 2. Team Logo Badge Injection
        var badge = pDoc.getElementById('vyoma-lunar-flux-badge');
        var logoB64 = "__TEAM_LOGO_B64__";

        if (!badge && logoB64 && logoB64.length > 50) {
            badge = pDoc.createElement('div');
            badge.id = 'vyoma-lunar-flux-badge';
            badge.innerHTML = '<img src="data:image/png;base64,' + logoB64 + '" style="width:100%;height:100%;object-fit:contain;" alt="Logo">';
            pDoc.body.appendChild(badge);
        }
    })();
    </script>
    """
    js_code = js_code.replace("__TEAM_LOGO_B64__", TEAM_LOGO_B64).replace("__LOCAL_VIDEO_B64__", LOCAL_VIDEO_B64)
    components.html(js_code, height=0, width=0)

def inject_youtube_background_video(video_id="pPuYfnaj_cc"):
    """
    Injects YouTube video (pPuYfnaj_cc) as a full-screen background video
    WITH 4,500 twinkling 3D starfield particles floating on top of it!
    """
    js_code = f"""
    <script>
    (function() {{
        var pWin = window.parent || window;
        var pDoc = pWin.document;

        // Force transparent backgrounds in parent document
        var styleId = 'vyoma-yt-bg-css';
        if (!pDoc.getElementById(styleId)) {{
            var style = pDoc.createElement('style');
            style.id = styleId;
            style.innerHTML = `
                html, body, .stApp, [data-testid="stAppViewContainer"], 
                [data-testid="stHeader"], header, [data-testid="stMain"], 
                .main, section.main, .stMainBlockContainer {{
                    background: transparent !important;
                    background-color: transparent !important;
                }}
                body {{
                    background-color: #020408 !important;
                }}
                #vyoma-yt-bg-video {{
                    position: fixed !important;
                    top: 50% !important;
                    left: 50% !important;
                    width: 177.77777778vh !important;
                    min-width: 100vw !important;
                    height: 56.25vw !important;
                    min-height: 100vh !important;
                    transform: translate(-50%, -50%) !important;
                    z-index: 1 !important;
                    pointer-events: none !important;
                    border: none !important;
                    filter: brightness(0.60) contrast(1.10) !important;
                }}
                [data-testid="stMain"], [data-testid="stSidebar"] {{
                    position: relative !important;
                    z-index: 10 !important;
                }}
            `;
            pDoc.head.appendChild(style);
        }}

        // Hide local video element if present
        var localVid = pDoc.getElementById('vyoma-local-bg-video');
        if (localVid) localVid.style.display = 'none';

        // Create or show YouTube Background iframe
        var iframe = pDoc.getElementById('vyoma-yt-bg-video');
        if (!iframe) {{
            iframe = pDoc.createElement('iframe');
            iframe.id = 'vyoma-yt-bg-video';
            iframe.src = 'https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&mute=1&loop=1&playlist={video_id}&controls=0&showinfo=0&autohide=1&modestbranding=1&rel=0&enablejsapi=1';
            iframe.allow = "autoplay; encrypted-media";
            pDoc.body.appendChild(iframe);
        }} else {{
            iframe.style.display = 'block';
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

# Always attach Gemini generated video as the primary background & Team LUNAR FLUX badge
inject_local_mp4_background_video()

# Page Routing System using session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "✨ VyomaSutra Homepage"

# Classic Glass Top Navigation Header
top_nav_col1, top_nav_col2 = st.columns([1.5, 1.2])

with top_nav_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 14px; padding: 4px 0;">
        <span style="font-size: 2.3rem; filter: drop-shadow(0 0 12px rgba(0, 163, 255, 0.9));">🌙</span>
        <div>
            <div style="font-size: 1.8rem; font-weight: 900; color: #FFFFFF; text-shadow: 0 0 16px rgba(0, 163, 255, 0.9); letter-spacing: 1px; line-height: 1.1;">
                VyomaSutra
            </div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.5px;">
                Autonomous Sub-Pixel Lunar Image Registration Engine &bull; SIH 2026 (PS 26166)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_nav_col2:
    nav_options = ["✨ VyomaSutra Homepage", "🚀 Command Center & Engine"]
    curr_index = 0 if st.session_state["current_page"] == "✨ VyomaSutra Homepage" else 1

    selected_top_mode = st.radio(
        "Top Navigation Mode",
        nav_options,
        index=curr_index,
        horizontal=True,
        label_visibility="collapsed",
        key="top_nav_radio_select"
    )
    if selected_top_mode != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_top_mode
        st.rerun()

st.markdown("""
<style>
/* Style Horizontal Top Navigation Radio Buttons into Futuristic Glass Pills */
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 12px !important;
    justify-content: flex-end !important;
    margin-top: 6px !important;
}
div[data-testid="stRadio"] label {
    background: rgba(15, 23, 42, 0.88) !important;
    border: 1.5px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 25px !important;
    padding: 10px 22px !important;
    cursor: pointer !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(14px) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6) !important;
}
div[data-testid="stRadio"] label:hover {
    border-color: #00A3FF !important;
    background: rgba(0, 163, 255, 0.2) !important;
    box-shadow: 0 0 20px rgba(0, 163, 255, 0.7) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stRadio"] label[data-checked="true"], 
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, #0284C7 0%, #00A3FF 100%) !important;
    border-color: #FFFFFF !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 22px rgba(0, 163, 255, 0.85) !important;
}
div[data-testid="stRadio"] input {
    display: none !important;
}
</style>
<hr style="border: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.5), transparent); margin: 15px 0 25px 0;">
""", unsafe_allow_html=True)

if st.session_state["current_page"] == "✨ VyomaSutra Homepage":
    st.markdown("""
    <style>
    .classic-hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(6, 11, 22, 0.92) 100%);
        border: 2px solid #00A3FF;
        border-radius: 20px;
        padding: 40px 35px;
        text-align: center;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-top: 5px;
        margin-bottom: 30px;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.95), 0 0 30px rgba(0, 163, 255, 0.45);
        position: relative;
        overflow: hidden;
    }
    .classic-hero-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 163, 255, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .classic-hero-card h1 {
        color: #FFFFFF !important;
        font-size: 3.2rem !important;
        font-weight: 900 !important;
        letter-spacing: 1.5px !important;
        margin-bottom: 10px !important;
        text-shadow: 0 0 22px rgba(0, 163, 255, 0.9) !important;
    }
    .classic-hero-card .subtitle {
        color: #38BDF8 !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        margin-bottom: 18px !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.6) !important;
    }
    .classic-hero-card .badge-tag {
        display: inline-block;
        background: rgba(0, 163, 255, 0.15);
        border: 1.5px solid #00A3FF;
        color: #FFFFFF;
        border-radius: 25px;
        padding: 8px 24px;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        box-shadow: 0 0 15px rgba(0, 163, 255, 0.3);
    }
    .metric-ribbon-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        backdrop-filter: blur(14px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.8);
    }
    .metric-ribbon-card .val {
        color: #38BDF8;
        font-size: 1.4rem;
        font-weight: 800;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.7);
    }
    .metric-ribbon-card .lbl {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .feature-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.90) 0%, rgba(11, 16, 29, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 16px;
        padding: 26px 22px;
        backdrop-filter: blur(16px);
        height: 100%;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.85);
        transition: all 0.35 ease;
    }
    .feature-card:hover {
        border-color: #00A3FF;
        box-shadow: 0 12px 35px rgba(0, 163, 255, 0.45);
        transform: translateY(-4px);
    }
    .feature-card h4 {
        color: #38BDF8 !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        margin-bottom: 10px !important;
    }
    .feature-card p {
        color: #CBD5E1 !important;
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
    }
    </style>

    <div class="classic-hero-card">
        <h1>🌙 VYOMASUTRA</h1>
        <div class="subtitle">Autonomous Sub-Pixel Lunar Image Registration Engine</div>
        <div class="badge-tag">SIH 2026 &bull; Problem Statement 26166 &bull; TEAM LUNAR FLUX</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric summary ribbon
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown("""
        <div class="metric-ribbon-card">
            <div class="val">&lt; 0.25 px</div>
            <div class="lbl">Reprojection RMSE Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="metric-ribbon-card">
            <div class="val">SuperPoint + LightGlue</div>
            <div class="lbl">Deep Neural Transformer Matcher</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="metric-ribbon-card">
            <div class="val">IAU 2000 Lunar CRS</div>
            <div class="lbl">GIS Ready Cartographic Export</div>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown("""
        <div class="metric-ribbon-card">
            <div class="val">100% Autonomous</div>
            <div class="lbl">Multi-Metric Risk Gatekeeper</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Feature grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Sub-Pixel Precision</h4>
            <p>Recovers homography alignment to <b>&lt; 0.25 px RMSE</b> using sub-pixel Taylor gradient expansion.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚡ Dual Matcher</h4>
            <p>Combines <b>Deep LightGlue + SuperPoint</b> with classical SIFT for high inlier yield in extreme shadows.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <h4>🗺️ GIS Product Delivery</h4>
            <p>Exports affine-aligned <b>GeoTIFF</b> rasters with true IAU 2000 Lunar Cartographic CRS metadata.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="feature-card">
            <h4>🛡️ Risk Gatekeeper</h4>
            <p>Autonomous multi-metric quality assessment evaluating SSIM, Distribution Score, and Illumination Risk.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ System Architecture Overview")
    
    st.markdown("""
    | Component Layer | Technology Stack | Scientific Target |
    | :--- | :--- | :--- |
    | **Preprocessing** | CLAHE + Resampling | Illumination equalization across extreme lunar crater shadows |
    | **Deep Feature Extraction** | SuperPoint + LightGlue | Transformer-based deep keypoint matching for low-texture polar terrain |
    | **Classical Baseline** | SIFT + RANSAC Filter | High-speed, robust geometric fallback |
    | **Sub-pixel Refinement** | Taylor Gradient Expansion | Sub-pixel accuracy down to < 0.25 px reprojection RMSE |
    | **GIS Cartographic Export** | Rasterio + GDAL GTiff | IAU 2000 Lunar Cartographic Reference System GeoTIFF delivery |
    """)

    st.markdown("---")
    st.write("")
    
    col_launch_btn, _ = st.columns([2, 1])
    with col_launch_btn:
        if st.button("🚀 Launch Command Center Engine", key="btn_enter_cmd_center"):
            st.session_state["current_page"] = "🚀 Command Center & Engine"
            st.rerun()

else:
    # -------------------------------------------------------------
    # PAGE 2: COMMAND CENTER & ENGINE
    # -------------------------------------------------------------
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(11, 16, 29, 0.88) 100%);
        border: 1.5px solid rgba(56, 189, 248, 0.45);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 22px;
        backdrop-filter: blur(18px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8), 0 0 20px rgba(56, 189, 248, 0.25);
    ">
        <div style="color: #38BDF8; font-size: 1.15rem; font-weight: 800; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            ⚙️ Pipeline Engine & Sensor Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns([1.2, 1.4, 1.4, 1.0])

    with cfg_c1:
        input_mode = st.selectbox(
            "Input Mode",
            ["🎯 Preloaded Lunar Demo Pair (Instant Demo)", "📂 Upload Custom TIFF Files"],
            index=0
        )

    with cfg_c2:
        sensor_pair = st.selectbox(
            "Sensor Pair Profile",
            [
                "OHRC — LROC NAC (High-Resolution Zoom: 0.25m)",
                "TMC-2 — LROC WAC (Terrain Mapping: 5m)",
                "IIRS — LROC WAC (Infrared Hyperspectral: 80m)"
            ],
            index=0
        )

    with cfg_c3:
        matcher_choice = st.selectbox(
            "Matching Engine",
            [
                "Dual Ensemble (SuperPoint+LightGlue + SIFT)",
                "Deep LightGlue + SuperPoint",
                "Classical SIFT Baseline"
            ],
            index=0
        )

    with cfg_c4:
        with st.popover("🛠️ Parameters", use_container_width=True):
            st.markdown("#### Advanced Settings")
            apply_clahe = st.checkbox("CLAHE Equalization", value=True, help="Equalizes extreme lunar shadow-contrast")
            enable_subpixel = st.checkbox("Sub-pixel Refinement", value=True, help="Refines detected points to sub-pixel coordinates")
            ransac_thresh = st.slider("RANSAC Thresh (px)", 1.0, 10.0, 3.0, 0.5)

    # Key mappings for backend
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

    # Load / Upload Data & Main View Columns
    source_array = None
    reference_array = None
    ref_crs = None
    ref_transform = None
    source_name = "Source Image"
    reference_name = "Reference Image"

    sample_dir = os.path.join(ROOT_DIR, "data", "samples")
src_sample_path = os.path.join(sample_dir, "ch2_ohr_ncp_20220914T0835371412_g_grd_d32.tif")
ref_sample_path = os.path.join(sample_dir, "ch2_ohr_ncp_20220914T1033119094_g_grd_d32.tif")
if not (os.path.exists(src_sample_path) and os.path.exists(ref_sample_path)):
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
        source_name = "Chandrayaan-2 OHRC Strip 1 (ch2_ohr_ncp_20220914T0835371412)"
        reference_name = "Chandrayaan-2 OHRC Strip 2 (ch2_ohr_ncp_20220914T1033119094)"
        st.markdown("""
        <style>
        .neon-demo-banner {
            background-color: rgba(20, 30, 50, 0.7);
            border: 1.5px solid #00A3FF;
            border-radius: 10px;
            padding: 16px 22px;
            margin-top: 15px;
            margin-bottom: 25px;
            color: #F8FAFC;
            font-family: 'Segoe UI', Roboto, sans-serif;
            font-size: 1.1rem;
            line-height: 1.5;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.8), 0 0 10px rgba(0, 163, 255, 0.25);
            transition: all 0.3s ease-in-out;
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .neon-demo-banner:hover {
            box-shadow: 0 0 25px rgba(0, 163, 255, 0.85), 0 0 10px rgba(56, 189, 248, 0.6);
            border-color: #38BDF8;
            background-color: rgba(25, 42, 70, 0.85);
            transform: translateY(-2px);
        }

        .neon-demo-banner .banner-icon {
            font-size: 1.6rem;
            filter: drop-shadow(0 0 8px rgba(255, 200, 0, 0.85));
            flex-shrink: 0;
        }

        .neon-demo-banner .banner-title {
            color: #38BDF8;
            font-weight: 800;
            margin-right: 6px;
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.7);
        }

        .neon-demo-banner .banner-desc {
            color: #FFFFFF;
            font-weight: 600;
        }
        </style>

        <div class="neon-demo-banner">
            <span class="banner-icon">💡</span>
            <div>
                <span class="banner-title">Demo Pair Loaded:</span>
                <span class="banner-desc">Chandrayaan-2 OHRC High-Resolution Lunar Crater Strips (083537 vs 103311) — 0.25m Spatial Resolution.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Sample pair not found on disk. Please upload custom files below.")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            s_file = st.file_uploader("Upload Moving / Source (.tif)", type=["tif", "tiff"])
        with col_up2:
            r_file = st.file_uploader("Upload Fixed / Reference (.tif)", type=["tif", "tiff"])
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

# Prominent Full-Width Primary Execution Button
run_pipeline_btn = st.button("🚀 Run Registration Pipeline", use_container_width=True)

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

        # Multi-Metric Validation Gatekeeper
        all_matches = result.get("matches", [])
        ransac_inliers = result.get("inliers", [])
        total_matches_count = len(all_matches)
        inliers_count = len(ransac_inliers)

        if total_matches_count == 0:
            st.error("⚠️ REGISTRATION FAILED: Zero matches found. Images are completely unrelated.")
            st.stop()

        inlier_ratio = inliers_count / total_matches_count if total_matches_count > 0 else 0.0

        if total_matches_count < 20 or inlier_ratio < 0.60:
            st.error("⚠️ REGISTRATION FAILED: Structurally unrelated images detected. Please ensure you uploaded overlapping lunar terrain.")
            st.warning(f"Diagnostics ➔ Total Matches: {total_matches_count} | Inlier Ratio: {inlier_ratio:.1%}")
            st.info("The pipeline requires at least 20 matches and a 60% inlier ratio to prevent severe map distortion. The warping phase has been aborted.")
            st.stop()

        if not result["success"]:
            st.error(f"❌ Alignment Failed: {result.get('error_message')}")
            st.stop()

        metrics = result.get("metrics", {})

        st.markdown(f"""
        <div style="
            background-color: #031c10 !important;
            background: #031c10 !important;
            border: 2px solid #10B981 !important;
            border-radius: 12px !important;
            padding: 14px 22px !important;
            margin-bottom: 12px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.98), 0 0 20px rgba(16, 185, 129, 0.4) !important;
            color: #FFFFFF !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        ">
            ✅ <strong style="color: #34D399 !important; font-size: 1.15rem !important; font-weight: 800 !important;">Validation Passed!</strong> 
            Inlier Ratio: {inlier_ratio:.1%}. Proceeding to warp...
        </div>
        <div style="
            background-color: #031c10 !important;
            background: #031c10 !important;
            border: 2px solid #10B981 !important;
            border-radius: 12px !important;
            padding: 14px 22px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.98), 0 0 20px rgba(16, 185, 129, 0.4) !important;
            color: #FFFFFF !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        ">
            ✅ <strong style="color: #34D399 !important; font-size: 1.15rem !important; font-weight: 800 !important;">Registration Succeeded</strong> in {result['elapsed_time_s']}s!
        </div>
        """, unsafe_allow_html=True)

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
            matches_to_plot = result["inliers"][:120]
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
            conf_emoji = {"HIGH": "🟢", "MODERATE": "LOW", "LOW": "🔴"}.get(conf_level, "⚪")
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
