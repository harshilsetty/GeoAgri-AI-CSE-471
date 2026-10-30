import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# 1. Initialize presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Color Palette definition
C_BG_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_BG_ALT = RGBColor(0xF6, 0xF9, 0xFC)
C_TEXT_NAVY = RGBColor(0x14, 0x2B, 0x45)
C_TEXT_SLATE = RGBColor(0x52, 0x65, 0x79)
C_TEAL = RGBColor(0x00, 0x9B, 0x8A)
C_SKY = RGBColor(0x39, 0x85, 0xD6)
C_AMBER = RGBColor(0xE9, 0xA2, 0x3B)
C_GREEN = RGBColor(0x23, 0x9B, 0x70)
C_CORAL = RGBColor(0xD9, 0x57, 0x57)
C_CARD_BG = RGBColor(0xF1, 0xF6, 0xF9)
C_BORDER = RGBColor(0xDC, 0xE5, 0xED)

FONT_HEADING = "Aptos"
FONT_BODY = "Inter"

def set_slide_background(slide, color=C_BG_WHITE):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_header(slide, title_text, category_text="CSE 471 — DEEP LEARNING FOR COMPUTER VISION"):
    # Category Tracker
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.name = FONT_BODY
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = C_TEAL

    # Title
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.7))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = FONT_HEADING
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = C_TEXT_NAVY

def add_footer(slide, slide_num, total_slides=14):
    f_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.35))
    tf = f_box.text_frame
    p = tf.paragraphs[0]
    p.text = f"GeoAgri-AI Research & Viva Presentation  |  Continuous Assessment (CA) Project  |  Slide {slide_num} of {total_slides}"
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = C_TEXT_SLATE

def create_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape

# ==========================================
# SLIDE 1: Cover
# ==========================================
slide1 = prs.slides.add_slide(blank_layout)
set_slide_background(slide1, C_BG_WHITE)

# Hero Header Tag
tag = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.0), Inches(3.2), Inches(0.4))
tag.fill.solid()
tag.fill.fore_color.rgb = RGBColor(0xE0, 0xF2, 0xF1)
tag.line.fill.background()
tf_tag = tag.text_frame
p_tag = tf_tag.paragraphs[0]
p_tag.text = "ACADEMIC CA PROJECT  •  SEM 5"
p_tag.font.name = FONT_BODY
p_tag.font.size = Pt(10)
p_tag.font.bold = True
p_tag.font.color.rgb = C_TEAL
p_tag.alignment = PP_ALIGN.CENTER

# Main Title
title_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.5), Inches(1.8))
tf1 = title_box.text_frame
tf1.word_wrap = True
p1 = tf1.paragraphs[0]
p1.text = "GEOAGRI-AI"
p1.font.name = FONT_HEADING
p1.font.size = Pt(46)
p1.font.bold = True
p1.font.color.rgb = C_TEXT_NAVY

p1_sub = tf1.add_paragraph()
p1_sub.text = "An Integrated Geospatial AI Decision-Support Framework for Agricultural Intelligence & Terrain Analysis"
p1_sub.font.name = FONT_BODY
p1_sub.font.size = Pt(17)
p1_sub.font.color.rgb = C_TEXT_SLATE

# 3 Interconnected Intelligence Cards
card_w = Inches(3.64)
card_h = Inches(2.7)

# Card 1: Crop
create_card(slide1, Inches(0.8), Inches(3.6), card_w, card_h)
tb_c1 = slide1.shapes.add_textbox(Inches(1.0), Inches(3.75), card_w - Inches(0.4), card_h - Inches(0.3))
tf_c1 = tb_c1.text_frame
tf_c1.word_wrap = True
p = tf_c1.paragraphs[0]
p.text = "🌱 CROP INTELLIGENCE"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_TEAL

items_c1 = [
    "• Model: XGBoost Multiclass Classifier",
    "• Scope: 26 agro-climatic features",
    "• Guardrails: FAO EcoCrop & ICAR physiological thresholds",
    "• Target: Climate-resilient crop selection"
]
for item in items_c1:
    p = tf_c1.add_paragraph()
    p.text = item
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_NAVY

# Card 2: Terrain
create_card(slide1, Inches(4.84), Inches(3.6), card_w, card_h)
tb_c2 = slide1.shapes.add_textbox(Inches(5.04), Inches(3.75), card_w - Inches(0.4), card_h - Inches(0.3))
tf_c2 = tb_c2.text_frame
tf_c2.word_wrap = True
p = tf_c2.paragraphs[0]
p.text = "🏔️ TERRAIN INTELLIGENCE"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_SKY

items_c2 = [
    "• Model: XGBoost Regressor / Classifier",
    "• Scope: 8 terrain & environmental features",
    "• Metrics: Slope, rainfall erosivity, soil type",
    "• Explainability: TreeSHAP local attributions"
]
for item in items_c2:
    p = tf_c2.add_paragraph()
    p.text = item
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_NAVY

# Card 3: Vision
create_card(slide1, Inches(8.88), Inches(3.6), card_w, card_h)
tb_c3 = slide1.shapes.add_textbox(Inches(9.08), Inches(3.75), card_w - Inches(0.4), card_h - Inches(0.3))
tf_c3 = tb_c3.text_frame
tf_c3.word_wrap = True
p = tf_c3.paragraphs[0]
p.text = "🛰️ VISION INTELLIGENCE"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_AMBER

items_c3 = [
    "• Model: YOLO11n-OBB (Oriented Boxes)",
    "• Scope: 5 archaeological & terrain classes",
    "• Resolution: 640 × 640 aerial imagery",
    "• Strategy: Damped class weighting for rare structures"
]
for item in items_c3:
    p = tf_c3.add_paragraph()
    p.text = item
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_NAVY

add_footer(slide1, 1)

# ==========================================
# SLIDE 2: Problem and Motivation
# ==========================================
slide2 = prs.slides.add_slide(blank_layout)
set_slide_background(slide2, C_BG_WHITE)
add_header(slide2, "One Landscape. Multiple Intelligence Needs.")

tb_sub2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.6))
p_sub2 = tb_sub2.text_frame.paragraphs[0]
p_sub2.text = "Agricultural landscapes operate under complex overlapping pedological, climatic, and structural constraints that fragmented single-purpose tools fail to address."
p_sub2.font.name = FONT_BODY
p_sub2.font.size = Pt(13)
p_sub2.font.color.rgb = C_TEXT_SLATE

card_w2 = Inches(3.64)
card_h2 = Inches(4.5)

# Card 1
create_card(slide2, Inches(0.8), Inches(2.2), card_w2, card_h2)
tb = slide2.shapes.add_textbox(Inches(1.0), Inches(2.4), card_w2 - Inches(0.4), card_h2 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "1. Agronomic Uncertainty"
p.font.name = FONT_HEADING
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = C_TEAL

p = tf.add_paragraph()
p.text = "Soil + Climate → Crop Selection\n"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

body_1 = [
    "• Traditional farmers rely on static historical tables that ignore micro-climate variability.",
    "• Multi-nutrient interactions (N, P, K, pH) and rainfall variance require non-linear modeling.",
    "• Pure ML models risk recommending crops that violate fundamental biological thresholds."
]
for b in body_1:
    p = tf.add_paragraph()
    p.text = b
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

# Card 2
create_card(slide2, Inches(4.84), Inches(2.2), card_w2, card_h2)
tb = slide2.shapes.add_textbox(Inches(5.04), Inches(2.4), card_w2 - Inches(0.4), card_h2 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "2. Land Degradation"
p.font.name = FONT_HEADING
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = C_SKY

p = tf.add_paragraph()
p.text = "Terrain + Rainfall → Soil Erosion\n"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

body_2 = [
    "• Steep slopes combined with intense monsoon rainfall accelerate topsoil loss and erosion.",
    "• Land-use planning requires continuous spatial risk assessment rather than regional averages.",
    "• Decisions must be explainable so land planners understand the underlying physical drivers."
]
for b in body_2:
    p = tf.add_paragraph()
    p.text = b
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

# Card 3
create_card(slide2, Inches(8.88), Inches(2.2), card_w2, card_h2)
tb = slide2.shapes.add_textbox(Inches(9.08), Inches(2.4), card_w2 - Inches(0.4), card_h2 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "3. Heritage & Physical Risk"
p.font.name = FONT_HEADING
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = C_AMBER

p = tf.add_paragraph()
p.text = "Aerial Imagery → Feature Detection\n"
p.font.name = FONT_BODY
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

body_3 = [
    "• Historic agrarian terrains contain ancient ruins, irrigation walls, and boundary structures.",
    "• Features lie at arbitrary rotational angles that traditional horizontal boxes fail to capture.",
    "• Unsupervised land development risks irreversible destruction of cultural heritage."
]
for b in body_3:
    p = tf.add_paragraph()
    p.text = b
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

add_footer(slide2, 2)

# ==========================================
# SLIDE 3: Research Gap
# ==========================================
slide3 = prs.slides.add_slide(blank_layout)
set_slide_background(slide3, C_BG_WHITE)
add_header(slide3, "Integration, Interpretability and Generalisation")

# Banner: Project Contribution
banner = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(11.733), Inches(0.75))
banner.fill.solid()
banner.fill.fore_color.rgb = RGBColor(0xEE, 0xF7, 0xF6)
banner.line.color.rgb = C_TEAL
banner.line.width = Pt(1)
tf_b = banner.text_frame
p_b = tf_b.paragraphs[0]
p_b.text = "CORE RESEARCH POSITION & PROJECT CONTRIBUTION"
p_b.font.name = FONT_HEADING
p_b.font.size = Pt(10)
p_b.font.bold = True
p_b.font.color.rgb = C_TEAL

p_b2 = tf_b.add_paragraph()
p_b2.text = "The contribution of GeoAgri-AI is system-level integration and rigorous empirical evaluation across heterogeneous modalities — not the speculative invention of a new ML algorithm."
p_b2.font.name = FONT_BODY
p_b2.font.size = Pt(12)
p_b2.font.color.rgb = C_TEXT_NAVY

# 3 Gaps
gap_w = Inches(3.64)
gap_h = Inches(4.1)

create_card(slide3, Inches(0.8), Inches(2.4), gap_w, gap_h)
tb = slide3.shapes.add_textbox(Inches(1.0), Inches(2.6), gap_w - Inches(0.4), gap_h - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "GAP 1: Fragmented Systems"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

items_g1 = [
    "• Existing agricultural tools treat crop prediction, terrain erosion, and aerial surveys as isolated silos.",
    "• No unified decision interface links subsurface soil chemistry directly to surface aerial imagery.",
    "• GeoAgri-AI Resolution: A single multi-tier platform harmonizing 26D agronomy, 8D terrain risk, and computer vision."
]
for item in items_g1:
    p = tf.add_paragraph()
    p.text = item
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

create_card(slide3, Inches(4.84), Inches(2.4), gap_w, gap_h)
tb = slide3.shapes.add_textbox(Inches(5.04), Inches(2.6), gap_w - Inches(0.4), gap_h - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "GAP 2: Interpretability Barrier"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

items_g2 = [
    "• Deep neural networks and ensemble trees act as 'black boxes', causing low user trust.",
    "• Pure ML models make high-confidence recommendations that violate physiological laws.",
    "• GeoAgri-AI Resolution: Exact TreeSHAP feature attribution coupled with ICAR/FAO physiological gating."
]
for item in items_g2:
    p = tf.add_paragraph()
    p.text = item
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

create_card(slide3, Inches(8.88), Inches(2.4), gap_w, gap_h)
tb = slide3.shapes.add_textbox(Inches(9.08), Inches(2.6), gap_w - Inches(0.4), gap_h - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "GAP 3: Generalisation Shift"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

items_g3 = [
    "• Aerial vision models tuned on one archaeological site fail completely on unseen geographies.",
    "• Background lithology shortcuts cause detectors to memorize local rocks rather than architecture.",
    "• GeoAgri-AI Resolution: Controlled empirical evaluation benchmarking domain augmentation and damped class weighting."
]
for item in items_g3:
    p = tf.add_paragraph()
    p.text = item
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

add_footer(slide3, 3)

# ==========================================
# SLIDE 4: Six-Layer Architecture
# ==========================================
slide4 = prs.slides.add_slide(blank_layout)
set_slide_background(slide4, C_BG_WHITE)
add_header(slide4, "Six Logical Layers Connect Data to Decisions")

layers = [
    ("LAYER 1: DATA ACQUISITION", "ISRIC SoilGrids (0-30cm N,P,K,pH)  •  Open-Meteo Climatology & Forecasts  •  SRTM 30m Elevation  •  Aerial Survey Imagery", C_CARD_BG, C_TEXT_NAVY),
    ("LAYER 2: DATA PROCESSING", "Multivariate Cleaning  •  Agronomic 26D Feature Engineering  •  OBB Coordinate Normalization  •  Spatial Indexing", C_CARD_BG, C_TEXT_NAVY),
    ("LAYER 3: ML INTELLIGENCE", "Crop: XGBoost (120 Trees)  •  Erosion: XGBoost (300 Trees)  •  Vision: YOLO11n-OBB (640x640, 196 Layers)", RGBColor(0xEE, 0xF7, 0xF6), C_TEAL),
    ("LAYER 4: BACKEND / API TIER", "FastAPI Asynchronous Gateway  •  Python Inference Bridge  •  JSON Protocol  •  Structured Error Handling", C_CARD_BG, C_TEXT_NAVY),
    ("LAYER 5: VISUALISATION ENGINE", "OpenCV OBB Polygon Overlays  •  Gaussian Feature Density Heatmap  •  Leaflet Geospatial Halos  •  TreeSHAP Plots", C_CARD_BG, C_TEXT_NAVY),
    ("LAYER 6: USER INTERFACE (UI)", "Next.js 14 App Router  •  TypeScript  •  Tailwind CSS  •  Interactive Map Navigation  •  Layer Toggles", RGBColor(0xEA, 0xF2, 0xFA), C_SKY),
]

layer_top = Inches(1.5)
layer_h = Inches(0.72)
layer_w = Inches(11.733)
spacing = Inches(0.18)

for i, (l_title, l_desc, bg_col, text_col) in enumerate(layers):
    curr_top = layer_top + i * (layer_h + spacing)
    card = create_card(slide4, Inches(0.8), curr_top, layer_w, layer_h, bg_col)
    
    tb = slide4.shapes.add_textbox(Inches(1.1), curr_top + Inches(0.08), layer_w - Inches(0.6), layer_h - Inches(0.16))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = l_title
    p.font.name = FONT_HEADING
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = text_col
    
    p2 = tf.add_paragraph()
    p2.text = l_desc
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = C_TEXT_SLATE

add_footer(slide4, 4)

# ==========================================
# SLIDE 5: Three AI Models
# ==========================================
slide5 = prs.slides.add_slide(blank_layout)
set_slide_background(slide5, C_BG_WHITE)
add_header(slide5, "Three Engines. Three Prediction Tasks.")

m_w = Inches(3.64)
m_h = Inches(5.1)

# Model 1
create_card(slide5, Inches(0.8), Inches(1.6), m_w, m_h)
tb = slide5.shapes.add_textbox(Inches(1.0), Inches(1.8), m_w - Inches(0.4), m_h - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "CROP INTELLIGENCE"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_TEAL

p_type = tf.add_paragraph()
p_type.text = "Task: Multiclass Classification\n"
p_type.font.name = FONT_BODY
p_type.font.size = Pt(11)
p_type.font.bold = True
p_type.font.color.rgb = C_TEXT_SLATE

specs_1 = [
    ("Algorithm", "XGBoost Classifier"),
    ("Features", "26 Engineered Agro-Climatic Features"),
    ("Inputs", "N, P, K, pH, rainfall, temp, slope"),
    ("Complexity", "120 Boosting Trees"),
    ("Gating", "FAO EcoCrop & ICAR Biophysical Rules"),
    ("Output", "Softmax Crop Suitability Slate"),
    ("Evaluation", "546 Chronological Holdout Queries")
]
for k, v in specs_1:
    p = tf.add_paragraph()
    p.text = f"• {k}: {v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_NAVY

# Model 2
create_card(slide5, Inches(4.84), Inches(1.6), m_w, m_h)
tb = slide5.shapes.add_textbox(Inches(5.04), Inches(1.8), m_w - Inches(0.4), m_h - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "TERRAIN EROSION"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_SKY

p_type = tf.add_paragraph()
p_type.text = "Task: Probability Regression & Risk Class\n"
p_type.font.name = FONT_BODY
p_type.font.size = Pt(11)
p_type.font.bold = True
p_type.font.color.rgb = C_TEXT_SLATE

specs_2 = [
    ("Algorithm", "XGBoost Classifier / Regressor"),
    ("Features", "8 Terrain & Environmental Features"),
    ("Inputs", "Slope, rainfall, soil, elevation, features"),
    ("Complexity", "300 Trees (Max Depth: 6)"),
    ("XAI Layer", "Exact TreeSHAP Shapley Values"),
    ("Output", "Risk Probability (0.0 to 1.0) & Class"),
    ("Risk Classes", "Low (<30%), Mod (30-69%), High (≥70%)")
]
for k, v in specs_2:
    p = tf.add_paragraph()
    p.text = f"• {k}: {v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_NAVY

# Model 3
create_card(slide5, Inches(8.88), Inches(1.6), m_w, m_h)
tb = slide5.shapes.add_textbox(Inches(9.08), Inches(1.8), m_w - Inches(0.4), m_h - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "AERIAL COMPUTER VISION"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_AMBER

p_type = tf.add_paragraph()
p_type.text = "Task: Oriented Object Detection (OBB)\n"
p_type.font.name = FONT_BODY
p_type.font.size = Pt(11)
p_type.font.bold = True
p_type.font.color.rgb = C_TEXT_SLATE

specs_3 = [
    ("Algorithm", "YOLO11n-OBB (Ultralytics)"),
    ("Layers", "196 Layers (344 PyTorch Modules)"),
    ("Parameters", "2,662,482 Parameters (2.66M)"),
    ("Input Size", "640 × 640 Aerial RGB Image"),
    ("Complexity", "6.7 GFLOPs (10.2 ms Latency)"),
    ("Loss Tuning", "Damped Inverse-Frequency Weights"),
    ("Output", "Rotated 4-pt Bounding Polygons & Conf")
]
for k, v in specs_3:
    p = tf.add_paragraph()
    p.text = f"• {k}: {v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_NAVY

add_footer(slide5, 5)

# ==========================================
# SLIDE 6: Dataset
# ==========================================
slide6 = prs.slides.add_slide(blank_layout)
set_slide_background(slide6, C_BG_WHITE)
add_header(slide6, "740 Aerial Images. 18,458 Annotations.")

# 4 KPI Cards
kpi_w = Inches(2.75)
kpi_h = Inches(1.3)
kpis = [
    ("740", "TOTAL AERIAL TILES", C_TEAL),
    ("18,458", "ORIENTED BOUNDING BOXES", C_SKY),
    ("5", "OBJECT CLASSES", C_AMBER),
    ("640 × 640", "IMAGE RESOLUTION", C_TEXT_NAVY)
]

for idx, (val, lbl, col) in enumerate(kpis):
    x = Inches(0.8) + idx * (kpi_w + Inches(0.24))
    create_card(slide6, x, Inches(1.5), kpi_w, kpi_h)
    tb = slide6.shapes.add_textbox(x, Inches(1.55), kpi_w, kpi_h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = val
    p.font.name = FONT_HEADING
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = col
    p.alignment = PP_ALIGN.CENTER
    
    p2 = tf.add_paragraph()
    p2.text = lbl
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_SLATE
    p2.alignment = PP_ALIGN.CENTER

# Two Columns: Splits & Classes
col_w = Inches(5.7)
col_h = Inches(3.6)

# Splits Card
create_card(slide6, Inches(0.8), Inches(3.1), col_w, col_h)
tb = slide6.shapes.add_textbox(Inches(1.0), Inches(3.25), col_w - Inches(0.4), col_h - Inches(0.3))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "DATASET SPLITS & SCENE DISTRIBUTION"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

splits = [
    ("Training Set (69.5%)", "514 images across 88 unique aerial scenes"),
    ("Validation Set (14.7%)", "109 images across 19 unique aerial scenes"),
    ("Held-Out Test Set (15.8%)", "117 images across 18 unique aerial scenes"),
]
for title, desc in splits:
    p = tf.add_paragraph()
    p.text = f"\n• {title}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p2 = tf.add_paragraph()
    p2.text = f"  {desc}"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = C_TEXT_SLATE

# Classes Card
create_card(slide6, Inches(6.833), Inches(3.1), col_w, col_h)
tb = slide6.shapes.add_textbox(Inches(7.033), Inches(3.25), col_w - Inches(0.4), col_h - Inches(0.3))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "CLASS DISTRIBUTION & SEVERE IMBALANCE"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

classes = [
    ("Boulders & Vegetation", "Abundant natural features spanning >80% of annotations"),
    ("Ruins & Others", "Moderate representation across historic site zones"),
    ("Structures (Critical Minority)", "Severely rare architectural class (<3% of labels). Models trained without weighting achieved 0.00% recall on held-out test data.")
]
for title, desc in classes:
    p = tf.add_paragraph()
    p.text = f"\n• {title}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_CORAL if "Structures" in title else C_SKY
    p2 = tf.add_paragraph()
    p2.text = f"  {desc}"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = C_TEXT_SLATE

add_footer(slide6, 6)

# ==========================================
# SLIDE 7: Experimental Comparison
# ==========================================
slide7 = prs.slides.add_slide(blank_layout)
set_slide_background(slide7, C_BG_WHITE)
add_header(slide7, "YOLO11 OBB Experimental Results")

# Table
rows = 5
cols = 6
left = Inches(0.8)
top = Inches(1.6)
width = Inches(11.733)
height = Inches(2.8)

table_shape = slide7.shapes.add_table(rows, cols, left, top, width, height)
table = table_shape.table

headers = ["Experiment", "Training Strategy", "Precision", "Recall", "mAP@50", "Structures Recall"]
for col_idx, h_text in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(0xEE, 0xF4, 0xF8)
    p = cell.text_frame.paragraphs[0]
    p.text = h_text
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_NAVY
    p.alignment = PP_ALIGN.CENTER

data = [
    ("EXP-B", "Geometry Refitting", "60.25% *", "37.31%", "43.06%", "63.89%"),
    ("EXP-C", "Domain Augmentation", "57.22%", "41.40%", "44.97% *", "72.22%"),
    ("EXP-D", "Damped Class Weights", "37.20%", "44.08% *", "41.50%", "77.11% *"),
    ("EXP-E", "Capacity Scaling (YOLO11s)", "46.80%", "37.17%", "37.97%", "55.56%"),
]

for row_idx, row_data in enumerate(data):
    for col_idx, val in enumerate(row_data):
        cell = table.cell(row_idx + 1, col_idx)
        cell.fill.solid()
        if "EXP-D" in row_data[0]:
            cell.fill.fore_color.rgb = RGBColor(0xF0, 0xF9, 0xF6)
        else:
            cell.fill.fore_color.rgb = C_BG_WHITE if row_idx % 2 == 0 else C_BG_ALT
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        if "*" in val:
            p.font.bold = True
            p.font.color.rgb = C_TEAL
        else:
            p.font.color.rgb = C_TEXT_NAVY
        p.alignment = PP_ALIGN.CENTER if col_idx > 1 else PP_ALIGN.LEFT

# Analysis Cards
c_w7 = Inches(3.7)
c_h7 = Inches(1.8)

create_card(slide7, Inches(0.8), Inches(4.8), c_w7, c_h7)
tb = slide7.shapes.add_textbox(Inches(0.95), Inches(4.9), c_w7 - Inches(0.3), c_h7)
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "EXP-B: Precision Peak (60.25%)"
p.font.name = FONT_HEADING
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY
p2 = tf.add_paragraph()
p2.text = "Rigorous polygon geometry refitting eliminated flipped normals and out-of-bounds annotations, producing the highest detection precision."
p2.font.name = FONT_BODY
p2.font.size = Pt(10)
p2.font.color.rgb = C_TEXT_SLATE

create_card(slide7, Inches(4.82), Inches(4.8), c_w7, c_h7)
tb = slide7.shapes.add_textbox(Inches(4.97), Inches(4.9), c_w7 - Inches(0.3), c_h7)
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "EXP-C: Overall mAP Peak (44.97%)"
p.font.name = FONT_HEADING
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_SKY
p2 = tf.add_paragraph()
p2.text = "HSV photometric jittering and 360-degree rotation invariance expanded domain coverage, achieving the highest overall mAP@50 (44.97%)."
p2.font.name = FONT_BODY
p2.font.size = Pt(10)
p2.font.color.rgb = C_TEXT_SLATE

create_card(slide7, Inches(8.83), Inches(4.8), c_w7, c_h7)
tb = slide7.shapes.add_textbox(Inches(8.98), Inches(4.9), c_w7 - Inches(0.3), c_h7)
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "EXP-D: Minority Recall Leader (77.11%)"
p.font.name = FONT_HEADING
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_TEAL
p2 = tf.add_paragraph()
p2.text = "Damped class weighting (structures penalty = 3.88x) maximized rare structure detection to 77.11% recall with 44.08% overall recall."
p2.font.name = FONT_BODY
p2.font.size = Pt(10)
p2.font.color.rgb = C_TEXT_SLATE

add_footer(slide7, 7)

# ==========================================
# SLIDE 8: Cross-Site Generalisation
# ==========================================
slide8 = prs.slides.add_slide(blank_layout)
set_slide_background(slide8, C_BG_WHITE)
add_header(slide8, "Validation Performance Did Not Transfer to Unseen Sites")

# Left Column: Recall Transfer Gap
create_card(slide8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.0))
tb = slide8.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(3.6))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "STRUCTURES RECALL TRANSFER FAILURE"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_CORAL

transfers = [
    ("EXP-C (Domain Augmentation)", "Validation: 72.22%", "Test Split: 0.00% (Complete Blackout)", C_CORAL),
    ("EXP-D (Class Weighted)", "Validation: 77.11%", "Test Split: 7.16% (Broke 0% Barrier)", C_TEAL),
]
for title, val_v, test_v, col in transfers:
    p = tf.add_paragraph()
    p.text = f"\n{title}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_NAVY
    
    p = tf.add_paragraph()
    p.text = f"  • {val_v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_SLATE
    
    p = tf.add_paragraph()
    p.text = f"  • {test_v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = col

# Right Column: EXP-D Test Metrics & Root Cause
create_card(slide8, Inches(6.933), Inches(1.5), Inches(5.6), Inches(4.0))
tb = slide8.shapes.add_textbox(Inches(7.133), Inches(1.7), Inches(5.2), Inches(3.6))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "EXP-D HELD-OUT TEST AUDIT & ROOT CAUSE"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

m_list = [
    "• Test Precision: 20.24%",
    "• Test Recall: 19.94%",
    "• Test mAP@50: 12.89%",
    "• Test mAP@50-95: 5.91%",
    "\nRoot Cause: Geographic Lithology Confounding",
    "In the existing dataset, boulders are 100% concentrated in Hampi granitic terrain, while structures reside in Nalanda/Pattadakal. Detectors learned local geological background shortcuts rather than transferable structural geometries."
]
for m in m_list:
    p = tf.add_paragraph()
    p.text = m
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_NAVY if "Test" in m else C_TEXT_SLATE

# Bottom Methodological Note
m_note = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.7), Inches(11.733), Inches(0.9))
m_note.fill.solid()
m_note.fill.fore_color.rgb = RGBColor(0xFD, 0xF6, 0xED)
m_note.line.color.rgb = C_AMBER
m_note.line.width = Pt(1)
tf_mn = m_note.text_frame
p_mn = tf_mn.paragraphs[0]
p_mn.text = "METHODOLOGICAL DISCLOSURE"
p_mn.font.name = FONT_HEADING
p_mn.font.size = Pt(10)
p_mn.font.bold = True
p_mn.font.color.rgb = C_AMBER

p_mn2 = tf_mn.add_paragraph()
p_mn2.text = "The test partition was consulted during earlier baseline audits. It serves here as an internal sanity check confirming cross-site degradation, not as an untouched, pristine external benchmark."
p_mn2.font.name = FONT_BODY
p_mn2.font.size = Pt(11)
p_mn2.font.color.rgb = C_TEXT_NAVY

add_footer(slide8, 8)

# ==========================================
# SLIDE 9: Model Architecture and Selection
# ==========================================
slide9 = prs.slides.add_slide(blank_layout)
set_slide_background(slide9, C_BG_WHITE)
add_header(slide9, "Why YOLO11n-OBB?")

# 3 Comparison Columns
c_w9 = Inches(3.64)
c_h9 = Inches(3.8)

# Col 1: YOLO11n
create_card(slide9, Inches(0.8), Inches(1.5), c_w9, c_h9, bg_color=RGBColor(0xF0, 0xF9, 0xF6), border_color=C_TEAL)
tb = slide9.shapes.add_textbox(Inches(1.0), Inches(1.7), c_w9 - Inches(0.4), c_h9 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "YOLO11n-OBB (EXP-D)"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_TEAL

p_sub = tf.add_paragraph()
p_sub.text = "Experimental Leader\n"
p_sub.font.name = FONT_BODY
p_sub.font.size = Pt(11)
p_sub.font.bold = True
p_sub.font.color.rgb = C_TEXT_SLATE

specs_9n = [
    ("Parameters", "2,662,482 (2.66M)"),
    ("Compute", "6.7 GFLOPs"),
    ("Latency", "10.2 ms / image"),
    ("Layers", "196 layers (344 modules)"),
    ("Structures Recall", "77.11% (Validation)"),
    ("Decision", "Selected as Research Leader")
]
for k, v in specs_9n:
    p = tf.add_paragraph()
    p.text = f"• {k}: {v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_NAVY

# Col 2: YOLO11s
create_card(slide9, Inches(4.84), Inches(1.5), c_w9, c_h9)
tb = slide9.shapes.add_textbox(Inches(5.04), Inches(1.7), c_w9 - Inches(0.4), c_h9 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "YOLO11s-OBB (EXP-E)"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_CORAL

p_sub = tf.add_paragraph()
p_sub.text = "Capacity Scaling Candidate\n"
p_sub.font.name = FONT_BODY
p_sub.font.size = Pt(11)
p_sub.font.bold = True
p_sub.font.color.rgb = C_TEXT_SLATE

specs_9s = [
    ("Parameters", "9,715,906 (9.72M - 3.6x)"),
    ("Compute", "22.5 GFLOPs (3.4x)"),
    ("Latency", "13.8 ms / image"),
    ("Layers", "196 layers (344 modules)"),
    ("Structures Recall", "55.56% (Overfit Drop)"),
    ("Decision", "Rejected (Overfit on 88 scenes)")
]
for k, v in specs_9s:
    p = tf.add_paragraph()
    p.text = f"• {k}: {v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_NAVY

# Col 3: YOLOv8s Prod
create_card(slide9, Inches(8.88), Inches(1.5), c_w9, c_h9)
tb = slide9.shapes.add_textbox(Inches(9.08), Inches(1.7), c_w9 - Inches(0.4), c_h9 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "YOLOv8s Baseline"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_SKY

p_sub = tf.add_paragraph()
p_sub.text = "Production Baseline\n"
p_sub.font.name = FONT_BODY
p_sub.font.size = Pt(11)
p_sub.font.bold = True
p_sub.font.color.rgb = C_TEXT_SLATE

specs_8 = [
    ("Parameters", "11,137,535 (11.14M)"),
    ("Compute", "28.7 GFLOPs"),
    ("Latency", "16.4 ms / image"),
    ("Layers", "129 layers (225 modules)"),
    ("Checkpoint", "runs/detect/yolov8s_archaeology2"),
    ("Decision", "Retained in Production")
]
for k, v in specs_8:
    p = tf.add_paragraph()
    p.text = f"• {k}: {v}"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_TEXT_NAVY

# Bottom Banner: Production Quarantine Policy
b_banner = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.5), Inches(11.733), Inches(1.1))
b_banner.fill.solid()
b_banner.fill.fore_color.rgb = RGBColor(0xF6, 0xF9, 0xFC)
b_banner.line.color.rgb = C_BORDER
b_banner.line.width = Pt(1)
tf_bb = b_banner.text_frame
p_bb = tf_bb.paragraphs[0]
p_bb.text = "PRODUCTION ISOLATION POLICY"
p_bb.font.name = FONT_HEADING
p_bb.font.size = Pt(10)
p_bb.font.bold = True
p_bb.font.color.rgb = C_TEXT_NAVY

p_bb2 = tf_bb.add_paragraph()
p_bb2.text = "EXP-D is recognized as the experimental research leader due to its superior minority-class recall (77.11%) and low computational footprint (6.7 GFLOPs). However, EXP-D is strictly quarantined in research and was NOT promoted to production in config/models_registry.yaml because cross-site generalization remains unproven."
p_bb2.font.name = FONT_BODY
p_bb2.font.size = Pt(10.5)
p_bb2.font.color.rgb = C_TEXT_SLATE

add_footer(slide9, 9)

# ==========================================
# SLIDE 10: Dashboard Visualisation
# ==========================================
slide10 = prs.slides.add_slide(blank_layout)
set_slide_background(slide10, C_BG_WHITE)
add_header(slide10, "Transparent Visual Analytics")

# 4 Dashboard Cards
c_w10 = Inches(2.75)
c_h10 = Inches(3.6)

v_cards = [
    ("1. OBB POLYGON FOOTPRINTS", "Semi-transparent colored polygons tightly enclosing rotated ruins and structures with class labels and confidence scores.", C_TEAL),
    ("2. FEATURE DENSITY HEATMAP", "Gaussian kernel density overlay centered at detected object centroids, exposing spatial clusters of heritage features.", C_SKY),
    ("3. EROSION RISK HEATMAP", "Multi-tier radial heat halos on the Leaflet GIS map dynamically colored by erosion probability (<30% Green, 30-69% Amber, ≥70% Red).", C_AMBER),
    ("4. INTERACTIVE LAYER TOGGLES", "Independent UI controls enabling users to toggle Heatmap Halos, Point Markers, and Bounding Overlays in real time.", C_TEXT_NAVY)
]

for idx, (title, desc, col) in enumerate(v_cards):
    x = Inches(0.8) + idx * (c_w10 + Inches(0.24))
    create_card(slide10, x, Inches(1.5), c_w10, c_h10)
    tb = slide10.shapes.add_textbox(x + Inches(0.15), Inches(1.7), c_w10 - Inches(0.3), c_h10 - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = FONT_HEADING
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = col
    
    p2 = tf.add_paragraph()
    p2.text = f"\n{desc}"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = C_TEXT_SLATE

# Prominent Academic Disclaimer Banner
d_banner = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.3))
d_banner.fill.solid()
d_banner.fill.fore_color.rgb = RGBColor(0xFE, 0xF3, 0xF2)
d_banner.line.color.rgb = C_CORAL
d_banner.line.width = Pt(1.5)
tf_db = d_banner.text_frame
p_db = tf_db.paragraphs[0]
p_db.text = "ACADEMIC VISUALIZATION INTEGRITY MANDATE"
p_db.font.name = FONT_HEADING
p_db.font.size = Pt(11)
p_db.font.bold = True
p_db.font.color.rgb = C_CORAL

p_db2 = tf_db.add_paragraph()
p_db2.text = '"Pixel-level segmentation is currently unavailable. The active computer vision model supports oriented object detection."\n\n' \
             'The dataset contains rotated bounding polygons, NOT pixel-level semantic masks. The GeoAgri-AI dashboard explicitly disclaims this and renders genuine OBB bounding footprints rather than fabricating fake semantic segmentation masks.'
p_db2.font.name = FONT_BODY
p_db2.font.size = Pt(10.5)
p_db2.font.color.rgb = C_TEXT_NAVY

add_footer(slide10, 10)

# ==========================================
# SLIDE 11: Technology Stack and Testing
# ==========================================
slide11 = prs.slides.add_slide(blank_layout)
set_slide_background(slide11, C_BG_WHITE)
add_header(slide11, "Full-Stack Integration and Verification")

# Stack Grid (Left)
create_card(slide11, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.1))
tb = slide11.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.7))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "MODERN DECOUPLED STACK"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

stack = [
    ("Frontend Tier", "Next.js 14, React 18, TypeScript, Tailwind CSS"),
    ("Backend API Tier", "FastAPI, Uvicorn, Python Prediction Bridge"),
    ("Machine Learning", "XGBoost, Ultralytics YOLO, PyTorch 2.6 CUDA"),
    ("Computer Vision", "OpenCV (cv2) polygon drawing, Gaussian blur"),
    ("Geospatial GIS", "Leaflet, React-Leaflet, Esri World Imagery"),
    ("Explainability", "TreeSHAP game-theory feature attributions"),
]
for title, tech in stack:
    p = tf.add_paragraph()
    p.text = f"\n• {title}:"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p2 = tf.add_paragraph()
    p2.text = f"  {tech}"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = C_TEXT_SLATE

# Verification Grid (Right)
create_card(slide11, Inches(6.933), Inches(1.5), Inches(5.6), Inches(5.1))
tb = slide11.shapes.add_textbox(Inches(7.133), Inches(1.7), Inches(5.2), Inches(4.7))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "REPORTED SOFTWARE TEST AUDIT"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_TEXT_NAVY

tests = [
    ("API Endpoint Suite", "4 / 4 PASSED (100%)", "Health, debug/model, crop recommendation, and point prediction endpoints verified in 61.11s.", C_GREEN),
    ("YOLO OBB Hardening Suite", "8 / 8 PASSED (100%)", "Rotated polygon bounding geometries, regression safety, and zero-box handling verified in 107.15s.", C_GREEN),
    ("Frontend ESLint Verification", "0 ERRORS (PASS)", "Strict Next.js 14 and React hooks static analysis passed clean.", C_GREEN),
    ("Production Next.js Build", "4 / 4 ROUTES COMPILED", "All static and dynamic routes (/, /_not-found, /api/predict) successfully bundled.", C_GREEN),
]
for title, status, desc, col in tests:
    p = tf.add_paragraph()
    p.text = f"\n• {title} — {status}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = col
    p2 = tf.add_paragraph()
    p2.text = f"  {desc}"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10)
    p2.font.color.rgb = C_TEXT_SLATE

add_footer(slide11, 11)

# ==========================================
# SLIDE 12: Limitations and Future Work
# ==========================================
slide12 = prs.slides.add_slide(blank_layout)
set_slide_background(slide12, C_BG_WHITE)
add_header(slide12, "Critical Reflection and Future Research")

c_w12 = Inches(5.6)
c_h12 = Inches(5.1)

# Limitations Column
create_card(slide12, Inches(0.8), Inches(1.5), c_w12, c_h12)
tb = slide12.shapes.add_textbox(Inches(1.0), Inches(1.7), c_w12 - Inches(0.4), c_h12 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "CURRENT SYSTEM LIMITATIONS"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_CORAL

limits = [
    ("Low Cross-Site Generalisation", "Detectors struggle when tested on completely unseen archaeological sites due to background rock shortcuts."),
    ("Extreme Class Imbalance", "Structures account for <3% of aerial annotations, causing high sensitivity to hyperparameter shifts."),
    ("Evaluation Integrity Scope", "The held-out test split was examined during baseline development; it serves as a sanity check, not a pristine external benchmark."),
    ("No Dense Pixel Masks", "The dataset is annotated with bounding polygons, not genuine semantic masks."),
    ("Crop & Erosion Formal Reports", "Formal standalone precision/recall reports require external field agronomic trials.")
]
for title, desc in limits:
    p = tf.add_paragraph()
    p.text = f"\n• {title}:"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_NAVY
    p2 = tf.add_paragraph()
    p2.text = f"  {desc}"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10)
    p2.font.color.rgb = C_TEXT_SLATE

# Future Work Column
create_card(slide12, Inches(6.933), Inches(1.5), c_w12, c_h12)
tb = slide12.shapes.add_textbox(Inches(7.133), Inches(1.7), c_w12 - Inches(0.4), c_h12 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "FUTURE RESEARCH ROADMAP"
p.font.name = FONT_HEADING
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_TEAL

futures = [
    ("AGB-120 Multi-Site Expansion", "Proposed 120-scene field collection across 6 new Indian archaeological sites to decouple structures from geology."),
    ("Untouched External Benchmark", "Strictly blind external test partition to objectively verify out-of-domain transfer."),
    ("Targeted Minority Re-Sampling", "Oversampling and tile-level mosaic augmentation focused strictly on structural features."),
    ("Dense Pixel Segmentation", "Curating polygon masks for U-Net / DeepLabV3+ pixel-level masonry boundary segmentation."),
    ("Multi-Site Agronomic Auditing", "Field sensor calibration across ICAR regional dryland research stations.")
]
for title, desc in futures:
    p = tf.add_paragraph()
    p.text = f"\n• {title}:"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_NAVY
    p2 = tf.add_paragraph()
    p2.text = f"  {desc}"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(10)
    p2.font.color.rgb = C_TEXT_SLATE

add_footer(slide12, 12)

# ==========================================
# SLIDE 13: Conclusion
# ==========================================
slide13 = prs.slides.add_slide(blank_layout)
set_slide_background(slide13, C_BG_WHITE)
add_header(slide13, "An Integrated Prototype. A Measurable Research Lesson.")

# 3 Takeaway Cards
t_w13 = Inches(3.64)
t_h13 = Inches(3.8)

create_card(slide13, Inches(0.8), Inches(1.5), t_w13, t_h13)
tb = slide13.shapes.add_textbox(Inches(1.0), Inches(1.7), t_w13 - Inches(0.4), t_h13 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "1. Engineering Integration"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_TEAL

body_c1 = [
    "• Successfully integrated 26D crop intelligence, 8D terrain erosion risk, and aerial OBB detection into a unified platform.",
    "• Connected Next.js 14 frontend with asynchronous FastAPI and in-memory PyTorch / XGBoost inference engines."
]
for b in body_c1:
    p = tf.add_paragraph()
    p.text = f"\n{b}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

create_card(slide13, Inches(4.84), Inches(1.5), t_w13, t_h13)
tb = slide13.shapes.add_textbox(Inches(5.04), Inches(1.7), t_w13 - Inches(0.4), t_h13 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "2. Empirical CV Trajectory"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_SKY

body_c2 = [
    "• Executed 4 controlled experiments (EXP-B through EXP-E) evaluating geometry refitting, domain augmentation, and class weighting.",
    "• Achieved 77.11% validation recall for rare structures via damped loss weighting ($w = 3.88x$) with 6.7 GFLOPs efficiency."
]
for b in body_c2:
    p = tf.add_paragraph()
    p.text = f"\n{b}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

create_card(slide13, Inches(8.88), Inches(1.5), t_w13, t_h13)
tb = slide13.shapes.add_textbox(Inches(9.08), Inches(1.7), t_w13 - Inches(0.4), t_h13 - Inches(0.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "3. Core Research Insight"
p.font.name = FONT_HEADING
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = C_AMBER

body_c3 = [
    "• Demonstrates that improving validation recall on a single-site dataset does NOT guarantee out-of-domain cross-site generalisation.",
    "• Established that capacity scaling (EXP-E) overfits on limited aerial scenes, confirming measured evidence over speculative scaling."
]
for b in body_c3:
    p = tf.add_paragraph()
    p.text = f"\n{b}"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = C_TEXT_SLATE

# Viva Discussion Banner
v_banner = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.5), Inches(11.733), Inches(1.1))
v_banner.fill.solid()
v_banner.fill.fore_color.rgb = RGBColor(0xEE, 0xF7, 0xF6)
v_banner.line.color.rgb = C_TEAL
v_banner.line.width = Pt(1.5)
tf_vb = v_banner.text_frame
p_vb = tf_vb.paragraphs[0]
p_vb.text = "THANK YOU — QUESTIONS & DEFENSE DISCUSSION"
p_vb.font.name = FONT_HEADING
p_vb.font.size = Pt(16)
p_vb.font.bold = True
p_vb.font.color.rgb = C_TEAL
p_vb.alignment = PP_ALIGN.CENTER

p_vb2 = tf_vb.add_paragraph()
p_vb2.text = "Department of Computer Science & Engineering  •  CSE 471: Deep Learning for Computer Vision"
p_vb2.font.name = FONT_BODY
p_vb2.font.size = Pt(11)
p_vb2.font.color.rgb = C_TEXT_SLATE
p_vb2.alignment = PP_ALIGN.CENTER

add_footer(slide13, 13)

# ==========================================
# SLIDE 14: References
# ==========================================
slide14 = prs.slides.add_slide(blank_layout)
set_slide_background(slide14, C_BG_WHITE)
add_header(slide14, "Selected Technical and Research References")

ref_box = slide14.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.1))
tf_ref = ref_box.text_frame
tf_ref.word_wrap = True

refs = [
    ("1. Archaeological Object Detection", "Olivier, M., & Verschoof-van der Vaart, W. B. (2021). Detecting Archaeological Features in Airborne Laser Scanning Data Using Automated Deep Learning. Journal of Computer Applications in Archaeology, 4(1), 1-18. DOI: 10.5334/jcaa.78"),
    ("2. Real-Time Object Detection Framework", "Jocher, G., Chaurasia, A., & Qiu, J. (2024). YOLO by Ultralytics (Version 11.0.0). Ultralytics Inc. Available at: https://github.com/ultralytics/ultralytics"),
    ("3. Gradient Boosted Tree Architecture", "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ACM."),
    ("4. Explainable AI & Feature Attribution", "Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1), 56-67."),
    ("5. Soil Pedology Database", "ISRIC — World Soil Information (2021). SoilGrids 250m 2.0: Global gridded soil information. Available at: https://soilgrids.org"),
    ("6. Meteorological Reanalysis & Forecasts", "Zippenfenig, P. (2023). Open-Meteo.com Weather API. Available at: https://open-meteo.com"),
    ("7. Academic Course Repository", "GeoAgri-AI — CSE 471 Deep Learning for Computer Vision Project Repository. Available at: https://github.com/harshilsetty/GeoAgri-AI-CSE-471")
]

for idx, (head, citation) in enumerate(refs):
    p = tf_ref.paragraphs[0] if idx == 0 else tf_ref.add_paragraph()
    p.text = f"{head}"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    
    p2 = tf_ref.add_paragraph()
    p2.text = f"{citation}\n"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = C_TEXT_SLATE

add_footer(slide14, 14)

# Output presentation
output_dir = "presentation"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "GeoAgri_AI_CSE471_Presentation.pptx")
prs.save(output_path)
print(f"SUCCESS: Presentation generated at: {output_path}")
