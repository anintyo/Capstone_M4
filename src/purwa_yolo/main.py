import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from PIL import Image 
from io import BytesIO
from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
import json

# PDF generation
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: fpdf2 not installed. PDF generation disabled.")

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ==================== SAFETY COMPLIANCE CONFIGURATION ====================

# Define PPE (Personal Protective Equipment) requirements
# IMPORTANT: Class names must match YOLO model output exactly
PPE_REQUIREMENTS = {
    'helmet': {
        'name_id': 'Helm Keselamatan',
        'priority': 'CRITICAL',
        'icon': '🪖',
        'description': 'Pelindung kepala dari benturan dan jatuhan benda'
    },
    'vest': {
        'name_id': 'Rompi Safety',
        'priority': 'CRITICAL', 
        'icon': '🦺',
        'description': 'Meningkatkan visibilitas pekerja'
    }
}

# Equipment categories untuk monitoring
EQUIPMENT_CATEGORIES = {
    'workers': ['person', 'worker', 'pekerja'],
    'heavy_equipment': ['excavator', 'crane', 'bulldozer', 'loader'],
    'tools': ['hammer', 'drill', 'saw', 'ladder'],
    'safety_barriers': ['cone', 'barrier', 'fence', 'warning_sign']
}

# Score thresholds
SCORE_THRESHOLDS = {
    'excellent': 90,
    'good': 75,
    'fair': 60,
    'poor': 0
}

# ==================== MODEL & ANNOTATOR CACHING ====================

@st.cache_resource
def load_model(model_name: str):
    """Load YOLO model with caching for performance"""
    model_path = MODELS_DIR / f"best_{model_name}.pt"
    if not model_path.exists():
        st.error(f"❌ Model tidak ditemukan: {model_path}")
        return None
    return YOLO(str(model_path))

@st.cache_resource
def get_annotators():
    """Get cached annotator objects"""
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
    return box_annotator, label_annotator

# ==================== SAFETY ANALYSIS FUNCTIONS ====================

def calculate_safety_score(classcounts: dict) -> dict:
    """
    Calculate comprehensive safety compliance score
    
    Args:
        classcounts: Dictionary of detected class names and their counts
    
    Returns:
        Dictionary containing safety score, compliance details, and recommendations
    """
    detected_classes = set(classcounts.keys())
    required_ppe = set(PPE_REQUIREMENTS.keys())
    
    # Find detected and missing PPE
    detected_ppe = detected_classes & required_ppe
    missing_ppe = required_ppe - detected_classes
    
    # Calculate weighted score based on priority
    priority_weights = {'CRITICAL': 40, 'HIGH': 30, 'MEDIUM': 20}
    total_possible_score = sum(priority_weights[PPE_REQUIREMENTS[ppe]['priority']] 
                               for ppe in required_ppe)
    
    achieved_score = sum(priority_weights[PPE_REQUIREMENTS[ppe]['priority']] 
                        for ppe in detected_ppe)
    
    compliance_percentage = (achieved_score / total_possible_score * 100) if total_possible_score > 0 else 0
    
    # Determine status
    if compliance_percentage >= SCORE_THRESHOLDS['excellent']:
        status = 'SANGAT BAIK'
        status_color = 'green'
        status_icon = '✅'
    elif compliance_percentage >= SCORE_THRESHOLDS['good']:
        status = 'BAIK'
        status_color = 'blue'
        status_icon = '👍'
    elif compliance_percentage >= SCORE_THRESHOLDS['fair']:
        status = 'CUKUP'
        status_color = 'orange'
        status_icon = '⚠️'
    else:
        status = 'KURANG'
        status_color = 'red'
        status_icon = '🚨'
    
    # Categorize missing PPE by priority
    missing_by_priority = {
        'CRITICAL': [],
        'HIGH': [],
        'MEDIUM': []
    }
    
    for ppe in missing_ppe:
        priority = PPE_REQUIREMENTS[ppe]['priority']
        missing_by_priority[priority].append(PPE_REQUIREMENTS[ppe])
    
    # Generate recommendations
    recommendations = generate_safety_recommendations(missing_by_priority, detected_ppe)
    
    return {
        'score': compliance_percentage,
        'status': status,
        'status_color': status_color,
        'status_icon': status_icon,
        'detected_ppe': list(detected_ppe),
        'missing_ppe': list(missing_ppe),
        'missing_by_priority': missing_by_priority,
        'detected_counts': classcounts,
        'recommendations': recommendations,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def generate_safety_recommendations(missing_by_priority: dict, detected_ppe: set) -> list:
    """Generate actionable safety recommendations"""
    recommendations = []
    
    # Critical missing items
    if missing_by_priority['CRITICAL']:
        for item in missing_by_priority['CRITICAL']:
            recommendations.append({
                'priority': 'CRITICAL',
                'message': f"SEGERA pastikan semua pekerja menggunakan {item['name_id']}",
                'icon': '🚨'
            })
    
    # High priority missing items
    if missing_by_priority['HIGH']:
        for item in missing_by_priority['HIGH']:
            recommendations.append({
                'priority': 'HIGH',
                'message': f"Periksa penggunaan {item['name_id']} di area kerja",
                'icon': '⚠️'
            })
    
    # Medium priority missing items
    if missing_by_priority['MEDIUM']:
        for item in missing_by_priority['MEDIUM']:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f"Disarankan menggunakan {item['name_id']} untuk keamanan maksimal",
                'icon': 'ℹ️'
            })
    
    # General recommendations
    if len(detected_ppe) > 0:
        recommendations.append({
            'priority': 'INFO',
            'message': f"Pertahankan penggunaan APD yang sudah terdeteksi",
            'icon': '👍'
        })
    
    return recommendations

def analyze_equipment_presence(classcounts: dict) -> dict:
    """Analyze workers, heavy equipment and tools presence"""
    analysis = {
        'workers': [],
        'heavy_equipment': [],
        'tools': [],
        'safety_barriers': [],
        'other': []
    }
    
    for class_name, count in classcounts.items():
        categorized = False
        for category, items in EQUIPMENT_CATEGORIES.items():
            if class_name in items:
                analysis[category].append({'name': class_name, 'count': count})
                categorized = True
                break
        
        if not categorized and class_name not in PPE_REQUIREMENTS:
            analysis['other'].append({'name': class_name, 'count': count})
    
    return analysis

# ==================== DETECTION PIPELINE ====================

def detector_pipeline_pillow(image_bytes, model):
    """Optimized detection pipeline with enhanced error handling"""
    try:
        # Load and convert image
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np_rgb = np.array(pil_image)
        
        # Run inference
        results = model(image_np_rgb, verbose=False)[0] 
        detections = sv.Detections.from_ultralytics(results).with_nms()
        
        # Get cached annotators
        box_annotator, label_annotator = get_annotators()
        
        # Annotate image
        annotated_image = pil_image.copy()
        annotated_image = box_annotator.annotate(scene=np.array(annotated_image), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Convert back to numpy array if needed
        if isinstance(annotated_image, Image.Image):
            annotated_image_np = np.asarray(annotated_image)
        else:
            annotated_image_np = annotated_image
        
        # Get class names and counts
        class_names = detections.data.get("class_name", [])
        classcounts = dict(Counter(class_names))
        
        return annotated_image_np, classcounts, detections
    
    except Exception as e:
        st.error(f"Error in detection pipeline: {str(e)}")
        return None, {}, None

# ==================== REPORT GENERATION ====================

# ==================== REPORT GENERATION ====================

def generate_pdf_report(safety_result: dict, image_name: str, annotated_image: np.ndarray = None) -> bytes:
    """
    Generate PDF report with format matching Ringkuman tab
    Returns PDF as bytes for download
    
    Note: Emojis/icons removed for font compatibility
    
    Args:
        safety_result: Safety analysis results
        image_name: Name of the image file
        annotated_image: Annotated image array (optional, will be included if provided)
    """
    if not PDF_AVAILABLE:
        raise ImportError("fpdf2 not installed. Install with: poetry install --extras 'pdf'")
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set margins
    pdf.set_margins(15, 15, 15)
    
    # ============ HEADER ============
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(220, 53, 69)  # Red color
    pdf.cell(0, 15, 'Safety Compliance Dashboard', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, 'Sistem Deteksi Kepatuhan APD: Helm & Rompi Safety', 0, 1, 'C')
    
    pdf.ln(5)
    
    # ============ METADATA ============
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 5, f'Tanggal: {safety_result["timestamp"]}', 0, 1)
    pdf.cell(0, 5, f'Gambar: {image_name}', 0, 1)
    
    pdf.ln(5)
    
    # ============ ANNOTATED IMAGE ============
    if annotated_image is not None:
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            import tempfile
            import os
            
            # Convert RGB numpy array to PIL Image
            if annotated_image.dtype != np.uint8:
                annotated_image = annotated_image.astype(np.uint8)
            
            pil_img = Image.fromarray(annotated_image)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_path = tmp_file.name
                pil_img.save(temp_path, 'JPEG', quality=85)
            
            # Calculate image dimensions to fit in PDF
            # Max width: 180mm (leaving margins)
            max_width = 180
            img_width = pil_img.width
            img_height = pil_img.height
            
            # Calculate scaled dimensions
            if img_width > img_height:
                # Landscape
                pdf_img_width = max_width
                pdf_img_height = (img_height / img_width) * max_width
            else:
                # Portrait - limit height too
                max_height = 100  # Max 100mm height
                scale = min(max_width / img_width, max_height / img_height)
                pdf_img_width = img_width * scale
                pdf_img_height = img_height * scale
            
            # Center the image
            x = (pdf.w - pdf_img_width) / 2
            
            # Add section title
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Gambar Hasil Deteksi:', 0, 1)
            pdf.ln(2)
            
            # Add image
            pdf.image(temp_path, x=x, y=pdf.get_y(), w=pdf_img_width, h=pdf_img_height)
            pdf.ln(pdf_img_height + 5)
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            # If image embedding fails, just skip it
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 5, f'[Gambar tidak dapat ditampilkan: {str(e)}]', 0, 1)
            pdf.ln(5)
    
    # ============ SCORE BOX ============
    # Determine color based on score
    if safety_result['score'] >= 90:
        bg_color = (40, 167, 69)  # Green
        status_symbol = '[EXCELLENT]'
    elif safety_result['score'] >= 75:
        bg_color = (0, 123, 255)  # Blue
        status_symbol = '[GOOD]'
    elif safety_result['score'] >= 60:
        bg_color = (255, 193, 7)  # Orange
        status_symbol = '[FAIR]'
    else:
        bg_color = (220, 53, 69)  # Red
        status_symbol = '[POOR]'
    
    # Draw score box
    pdf.set_fill_color(*bg_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 32)
    
    # Centered box
    box_width = 180
    box_height = 30
    x = (pdf.w - box_width) / 2
    
    pdf.set_xy(x, pdf.get_y())
    pdf.cell(box_width, box_height, '', 0, 0, 'C', fill=True)
    
    # Score text (without emoji)
    pdf.set_xy(x, pdf.get_y() + 5)
    pdf.cell(box_width, 10, f'{safety_result["score"]:.1f}%', 0, 0, 'C')
    
    # Status text
    pdf.set_font('Arial', 'B', 16)
    pdf.set_xy(x, pdf.get_y() + 12)
    pdf.cell(box_width, 10, safety_result['status'], 0, 1, 'C')
    
    pdf.ln(10)
    
    # ============ SUMMARY METRICS ============
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 11)
    
    # Metrics boxes
    col_width = 85
    
    # APD Terdeteksi
    pdf.set_fill_color(220, 252, 231)  # Light green
    pdf.cell(col_width, 10, f'APD Terdeteksi: {len(safety_result["detected_ppe"])}', 1, 0, 'C', fill=True)
    
    pdf.cell(10)  # Spacing
    
    # APD Kurang
    pdf.set_fill_color(255, 229, 229)  # Light red
    pdf.cell(col_width, 10, f'APD Kurang: {len(safety_result["missing_ppe"])}', 1, 1, 'C', fill=True)
    
    pdf.ln(10)
    
    # ============ RINGKASAN KEPATUHAN APD ============
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Ringkasan Kepatuhan APD', 0, 1)
    
    pdf.ln(2)
    
    # Table header
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(200, 200, 200)
    
    col_widths = [30, 80, 40, 30]
    headers = ['Status', 'APD', 'Prioritas', 'Jumlah']
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, 'C', fill=True)
    pdf.ln()
    
    # Table rows - Detected PPE
    pdf.set_font('Arial', '', 9)
    pdf.set_fill_color(240, 255, 240)  # Very light green
    
    for ppe in safety_result['detected_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        
        # Status
        pdf.set_text_color(0, 128, 0)  # Green
        pdf.cell(col_widths[0], 7, 'Terdeteksi', 1, 0, 'C', fill=True)
        
        # APD Name (without emoji)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[1], 7, ppe_info.get('name_id', ppe), 1, 0, 'L', fill=True)
        
        # Priority
        priority = ppe_info.get('priority', 'N/A')
        if priority == 'CRITICAL':
            pdf.set_text_color(220, 53, 69)
        elif priority == 'HIGH':
            pdf.set_text_color(255, 193, 7)
        else:
            pdf.set_text_color(0, 123, 255)
        pdf.cell(col_widths[2], 7, priority, 1, 0, 'C', fill=True)
        
        # Count
        pdf.set_text_color(0, 0, 0)
        count = safety_result['detected_counts'].get(ppe, 0)
        pdf.cell(col_widths[3], 7, str(count), 1, 1, 'C', fill=True)
    
    # Table rows - Missing PPE
    pdf.set_fill_color(255, 240, 240)  # Very light red
    
    for ppe in safety_result['missing_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        
        # Status
        pdf.set_text_color(220, 53, 69)  # Red
        pdf.cell(col_widths[0], 7, 'Kurang', 1, 0, 'C', fill=True)
        
        # APD Name (without emoji)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[1], 7, ppe_info.get('name_id', ppe), 1, 0, 'L', fill=True)
        
        # Priority
        priority = ppe_info.get('priority', 'N/A')
        if priority == 'CRITICAL':
            pdf.set_text_color(220, 53, 69)
        elif priority == 'HIGH':
            pdf.set_text_color(255, 193, 7)
        else:
            pdf.set_text_color(0, 123, 255)
        pdf.cell(col_widths[2], 7, priority, 1, 0, 'C', fill=True)
        
        # Count
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[3], 7, '0', 1, 1, 'C', fill=True)
    
    pdf.ln(10)
    
    # ============ DETAIL BREAKDOWN ============
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Detail Kepatuhan', 0, 1)
    
    # APD Terdeteksi
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 6, '[OK] APD Terdeteksi:', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    
    if safety_result['detected_ppe']:
        for ppe in safety_result['detected_ppe']:
            ppe_info = PPE_REQUIREMENTS.get(ppe, {})
            count = safety_result['detected_counts'].get(ppe, 0)
            pdf.cell(10)  # Indent
            # Remove emoji icon, use bullet point
            pdf.cell(0, 5, f'  - {ppe_info.get("name_id", ppe)}: {count} unit', 0, 1)
    else:
        pdf.cell(10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, 'Tidak ada APD terdeteksi', 0, 1)
    
    pdf.ln(3)
    
    # APD Tidak Terdeteksi
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(220, 53, 69)
    pdf.cell(0, 6, '[!] APD Tidak Terdeteksi:', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    
    if safety_result['missing_ppe']:
        for ppe in safety_result['missing_ppe']:
            ppe_info = PPE_REQUIREMENTS.get(ppe, {})
            priority = ppe_info.get('priority', 'N/A')
            
            pdf.cell(10)  # Indent
            
            # Add priority indicator
            if priority == 'CRITICAL':
                priority_text = ' [CRITICAL]'
                pdf.set_text_color(220, 53, 69)
            elif priority == 'HIGH':
                priority_text = ' [HIGH]'
                pdf.set_text_color(255, 193, 7)
            else:
                priority_text = ''
                pdf.set_text_color(0, 0, 0)
            
            # Remove emoji, use bullet
            pdf.cell(0, 5, f'  - {ppe_info.get("name_id", ppe)}{priority_text}', 0, 1)
    else:
        pdf.cell(10)
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 5, 'Semua APD terdeteksi!', 0, 1)
    
    pdf.ln(8)
    
    # ============ REKOMENDASI ============
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Rekomendasi Keselamatan', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    
    if safety_result['recommendations']:
        for i, rec in enumerate(safety_result['recommendations'], 1):
            # Priority color
            if rec['priority'] == 'CRITICAL':
                pdf.set_text_color(220, 53, 69)
                priority_label = '[CRITICAL] '
            elif rec['priority'] == 'HIGH':
                pdf.set_text_color(255, 140, 0)
                priority_label = '[HIGH] '
            elif rec['priority'] == 'MEDIUM':
                pdf.set_text_color(0, 123, 255)
                priority_label = '[MEDIUM] '
            else:
                pdf.set_text_color(0, 0, 0)
                priority_label = ''
            
            # Recommendation text with multi_cell for wrapping
            pdf.cell(10)  # Indent
            pdf.multi_cell(0, 5, f"{i}. {priority_label}{rec['message']}")
            pdf.ln(1)
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 5, '[OK] Tidak ada rekomendasi - Kepatuhan sempurna!', 0, 1)
    
    pdf.ln(10)
    
    # ============ FOOTER ============
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, 'Laporan dibuat secara otomatis oleh Safety Compliance Dashboard', 0, 1, 'C')
    pdf.cell(0, 5, 'Sistem Deteksi Keselamatan Konstruksi berbasis AI', 0, 1, 'C')
    
    # Return PDF as bytes (output() already returns bytes, no need to encode)
    return bytes(pdf.output())
    """
    Generate PDF report with format matching Ringkuman tab
    Returns PDF as bytes for download
    
    Note: Emojis/icons removed for font compatibility
    """
    if not PDF_AVAILABLE:
        raise ImportError("fpdf2 not installed. Install with: poetry install --extras 'pdf'")
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set margins
    pdf.set_margins(15, 15, 15)
    
    # ============ HEADER ============
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(220, 53, 69)  # Red color
    pdf.cell(0, 15, 'Safety Compliance Dashboard', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, 'Sistem Deteksi Kepatuhan APD: Helm & Rompi Safety', 0, 1, 'C')
    
    pdf.ln(5)
    
    # ============ METADATA ============
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 5, f'Tanggal: {safety_result["timestamp"]}', 0, 1)
    pdf.cell(0, 5, f'Gambar: {image_name}', 0, 1)
    
    pdf.ln(5)
    
    # ============ SCORE BOX ============
    # Determine color based on score
    if safety_result['score'] >= 90:
        bg_color = (40, 167, 69)  # Green
        status_symbol = '[EXCELLENT]'
    elif safety_result['score'] >= 75:
        bg_color = (0, 123, 255)  # Blue
        status_symbol = '[GOOD]'
    elif safety_result['score'] >= 60:
        bg_color = (255, 193, 7)  # Orange
        status_symbol = '[FAIR]'
    else:
        bg_color = (220, 53, 69)  # Red
        status_symbol = '[POOR]'
    
    # Draw score box
    pdf.set_fill_color(*bg_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 32)
    
    # Centered box
    box_width = 180
    box_height = 30
    x = (pdf.w - box_width) / 2
    
    pdf.set_xy(x, pdf.get_y())
    pdf.cell(box_width, box_height, '', 0, 0, 'C', fill=True)
    
    # Score text (without emoji)
    pdf.set_xy(x, pdf.get_y() + 5)
    pdf.cell(box_width, 10, f'{safety_result["score"]:.1f}%', 0, 0, 'C')
    
    # Status text
    pdf.set_font('Arial', 'B', 16)
    pdf.set_xy(x, pdf.get_y() + 12)
    pdf.cell(box_width, 10, safety_result['status'], 0, 1, 'C')
    
    pdf.ln(10)
    
    # ============ SUMMARY METRICS ============
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 11)
    
    # Metrics boxes
    col_width = 85
    
    # APD Terdeteksi
    pdf.set_fill_color(220, 252, 231)  # Light green
    pdf.cell(col_width, 10, f'APD Terdeteksi: {len(safety_result["detected_ppe"])}', 1, 0, 'C', fill=True)
    
    pdf.cell(10)  # Spacing
    
    # APD Kurang
    pdf.set_fill_color(255, 229, 229)  # Light red
    pdf.cell(col_width, 10, f'APD Kurang: {len(safety_result["missing_ppe"])}', 1, 1, 'C', fill=True)
    
    pdf.ln(10)
    
    # ============ RINGKASAN KEPATUHAN APD ============
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Ringkasan Kepatuhan APD', 0, 1)
    
    pdf.ln(2)
    
    # Table header
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(200, 200, 200)
    
    col_widths = [30, 80, 40, 30]
    headers = ['Status', 'APD', 'Prioritas', 'Jumlah']
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, 'C', fill=True)
    pdf.ln()
    
    # Table rows - Detected PPE
    pdf.set_font('Arial', '', 9)
    pdf.set_fill_color(240, 255, 240)  # Very light green
    
    for ppe in safety_result['detected_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        
        # Status
        pdf.set_text_color(0, 128, 0)  # Green
        pdf.cell(col_widths[0], 7, 'Terdeteksi', 1, 0, 'C', fill=True)
        
        # APD Name (without emoji)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[1], 7, ppe_info.get('name_id', ppe), 1, 0, 'L', fill=True)
        
        # Priority
        priority = ppe_info.get('priority', 'N/A')
        if priority == 'CRITICAL':
            pdf.set_text_color(220, 53, 69)
        elif priority == 'HIGH':
            pdf.set_text_color(255, 193, 7)
        else:
            pdf.set_text_color(0, 123, 255)
        pdf.cell(col_widths[2], 7, priority, 1, 0, 'C', fill=True)
        
        # Count
        pdf.set_text_color(0, 0, 0)
        count = safety_result['detected_counts'].get(ppe, 0)
        pdf.cell(col_widths[3], 7, str(count), 1, 1, 'C', fill=True)
    
    # Table rows - Missing PPE
    pdf.set_fill_color(255, 240, 240)  # Very light red
    
    for ppe in safety_result['missing_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        
        # Status
        pdf.set_text_color(220, 53, 69)  # Red
        pdf.cell(col_widths[0], 7, 'Kurang', 1, 0, 'C', fill=True)
        
        # APD Name (without emoji)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[1], 7, ppe_info.get('name_id', ppe), 1, 0, 'L', fill=True)
        
        # Priority
        priority = ppe_info.get('priority', 'N/A')
        if priority == 'CRITICAL':
            pdf.set_text_color(220, 53, 69)
        elif priority == 'HIGH':
            pdf.set_text_color(255, 193, 7)
        else:
            pdf.set_text_color(0, 123, 255)
        pdf.cell(col_widths[2], 7, priority, 1, 0, 'C', fill=True)
        
        # Count
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[3], 7, '0', 1, 1, 'C', fill=True)
    
    pdf.ln(10)
    
    # ============ DETAIL BREAKDOWN ============
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Detail Kepatuhan', 0, 1)
    
    # APD Terdeteksi
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 6, '[OK] APD Terdeteksi:', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    
    if safety_result['detected_ppe']:
        for ppe in safety_result['detected_ppe']:
            ppe_info = PPE_REQUIREMENTS.get(ppe, {})
            count = safety_result['detected_counts'].get(ppe, 0)
            pdf.cell(10)  # Indent
            # Remove emoji icon, use bullet point
            pdf.cell(0, 5, f'  - {ppe_info.get("name_id", ppe)}: {count} unit', 0, 1)
    else:
        pdf.cell(10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, 'Tidak ada APD terdeteksi', 0, 1)
    
    pdf.ln(3)
    
    # APD Tidak Terdeteksi
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(220, 53, 69)
    pdf.cell(0, 6, '[!] APD Tidak Terdeteksi:', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    
    if safety_result['missing_ppe']:
        for ppe in safety_result['missing_ppe']:
            ppe_info = PPE_REQUIREMENTS.get(ppe, {})
            priority = ppe_info.get('priority', 'N/A')
            
            pdf.cell(10)  # Indent
            
            # Add priority indicator
            if priority == 'CRITICAL':
                priority_text = ' [CRITICAL]'
                pdf.set_text_color(220, 53, 69)
            elif priority == 'HIGH':
                priority_text = ' [HIGH]'
                pdf.set_text_color(255, 193, 7)
            else:
                priority_text = ''
                pdf.set_text_color(0, 0, 0)
            
            # Remove emoji, use bullet
            pdf.cell(0, 5, f'  - {ppe_info.get("name_id", ppe)}{priority_text}', 0, 1)
    else:
        pdf.cell(10)
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 5, 'Semua APD terdeteksi!', 0, 1)
    
    pdf.ln(8)
    
    # ============ REKOMENDASI ============
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Rekomendasi Keselamatan', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    
    if safety_result['recommendations']:
        for i, rec in enumerate(safety_result['recommendations'], 1):
            # Priority color
            if rec['priority'] == 'CRITICAL':
                pdf.set_text_color(220, 53, 69)
                priority_label = '[CRITICAL] '
            elif rec['priority'] == 'HIGH':
                pdf.set_text_color(255, 140, 0)
                priority_label = '[HIGH] '
            elif rec['priority'] == 'MEDIUM':
                pdf.set_text_color(0, 123, 255)
                priority_label = '[MEDIUM] '
            else:
                pdf.set_text_color(0, 0, 0)
                priority_label = ''
            
            # Recommendation text with multi_cell for wrapping
            pdf.cell(10)  # Indent
            pdf.multi_cell(0, 5, f"{i}. {priority_label}{rec['message']}")
            pdf.ln(1)
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 5, '[OK] Tidak ada rekomendasi - Kepatuhan sempurna!', 0, 1)
    
    pdf.ln(10)
    
    # ============ FOOTER ============
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, 'Laporan dibuat secara otomatis oleh Safety Compliance Dashboard', 0, 1, 'C')
    pdf.cell(0, 5, 'Sistem Deteksi Keselamatan Konstruksi berbasis AI', 0, 1, 'C')
    
    # Return PDF as bytes (output() already returns bytes, no need to encode)
    return bytes(pdf.output())


def save_compliance_report(safety_result: dict, image_name: str, annotated_image: np.ndarray = None):
    """
    Save compliance report and return PDF data
    
    Args:
        safety_result: Safety analysis results
        image_name: Name of the image file
        annotated_image: Annotated image with bounding boxes (optional)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"safety_report_{timestamp}.pdf"
    
    # Generate PDF with image
    try:
        pdf_bytes = generate_pdf_report(safety_result, image_name, annotated_image)
        
        # Try to save to reports directory if it exists
        try:
            report_path = REPORTS_DIR / report_filename
            with open(report_path, 'wb') as f:
                f.write(pdf_bytes)
            saved_path = str(report_path)
        except Exception as e:
            # If fails, just return the data without saving to disk
            saved_path = None
        
        return pdf_bytes, report_filename, saved_path
        
    except ImportError as e:
        # If fpdf2 not installed, create simple text report
        st.error("⚠️ PDF library tidak terinstall. Install dengan: poetry install --extras 'pdf'")
        return None, None, None

def generate_report_summary(safety_result: dict) -> pd.DataFrame:
    """Generate summary table for display (used in Ringkasan tab)"""
    data = []
    
    # Add detected PPE
    for ppe in safety_result['detected_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        data.append({
            'Status': '✅ Terdeteksi',
            'APD': ppe_info.get('name_id', ppe),
            'Prioritas': ppe_info.get('priority', 'N/A'),
            'Jumlah': safety_result['detected_counts'].get(ppe, 0)
        })
    
    # Add missing PPE
    for ppe in safety_result['missing_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        data.append({
            'Status': '❌ Tidak Terdeteksi',
            'APD': ppe_info.get('name_id', ppe),
            'Prioritas': ppe_info.get('priority', 'N/A'),
            'Jumlah': 0
        })
    
    return pd.DataFrame(data)

# ==================== UI COMPONENTS ====================

def display_safety_dashboard(safety_result: dict, annotated_image: np.ndarray = None):
    """Display comprehensive safety compliance dashboard
    
    Args:
        safety_result: Safety analysis results
        annotated_image: Annotated image with detection boxes (optional, for PDF report)
    """
    
    st.markdown("---")
    st.header("🛡️ Safety Compliance Dashboard")
    
    # Score Display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style='padding: 20px; background-color: {safety_result['status_color']}; 
        border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>{safety_result['status_icon']} {safety_result['score']:.1f}%</h1>
            <h3 style='color: white; margin: 0;'>{safety_result['status']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("APD Terdeteksi", len(safety_result['detected_ppe']))
    
    with col3:
        st.metric("APD Kurang", len(safety_result['missing_ppe']))
    
    st.markdown("---")
    
    # Detailed Analysis in Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan", 
        "🔍 Detail APD", 
        "💡 Rekomendasi",
        "📄 Laporan"
    ])
    
    with tab1:
        st.subheader("Ringkasan Kepatuhan APD")
        summary_df = generate_report_summary(safety_result)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Visual breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ APD Terdeteksi:**")
            if safety_result['detected_ppe']:
                for ppe in safety_result['detected_ppe']:
                    ppe_info = PPE_REQUIREMENTS.get(ppe, {})
                    count = safety_result['detected_counts'].get(ppe, 0)
                    st.success(f"{ppe_info.get('icon', '•')} {ppe_info.get('name_id', ppe)}: **{count}** unit")
            else:
                st.info("Tidak ada APD terdeteksi")
        
        with col2:
            st.markdown("**❌ APD Tidak Terdeteksi:**")
            if safety_result['missing_ppe']:
                for ppe in safety_result['missing_ppe']:
                    ppe_info = PPE_REQUIREMENTS.get(ppe, {})
                    priority = ppe_info.get('priority', 'N/A')
                    if priority == 'CRITICAL':
                        st.error(f"{ppe_info.get('icon', '•')} {ppe_info.get('name_id', ppe)} ⚠️ CRITICAL")
                    elif priority == 'HIGH':
                        st.warning(f"{ppe_info.get('icon', '•')} {ppe_info.get('name_id', ppe)}")
                    else:
                        st.info(f"{ppe_info.get('icon', '•')} {ppe_info.get('name_id', ppe)}")
            else:
                st.success("Semua APD terdeteksi!")
    
    with tab2:
        st.subheader("Detail Alat Pelindung Diri (APD)")
        
        for ppe_key, ppe_info in PPE_REQUIREMENTS.items():
            with st.expander(f"{ppe_info['icon']} {ppe_info['name_id']} - {ppe_info['priority']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Deskripsi:** {ppe_info['description']}")
                    if ppe_key in safety_result['detected_ppe']:
                        count = safety_result['detected_counts'].get(ppe_key, 0)
                        st.success(f"✅ Status: Terdeteksi ({count} unit)")
                    else:
                        st.error(f"❌ Status: Tidak terdeteksi")
                with col2:
                    st.write(f"**Prioritas:**")
                    if ppe_info['priority'] == 'CRITICAL':
                        st.error(ppe_info['priority'])
                    elif ppe_info['priority'] == 'HIGH':
                        st.warning(ppe_info['priority'])
                    else:
                        st.info(ppe_info['priority'])
    
    with tab3:
        st.subheader("💡 Rekomendasi Keselamatan")
        
        if safety_result['recommendations']:
            for i, rec in enumerate(safety_result['recommendations'], 1):
                if rec['priority'] == 'CRITICAL':
                    st.error(f"{rec['icon']} **{rec['priority']}:** {rec['message']}")
                elif rec['priority'] == 'HIGH':
                    st.warning(f"{rec['icon']} **{rec['priority']}:** {rec['message']}")
                elif rec['priority'] == 'MEDIUM':
                    st.info(f"{rec['icon']} **{rec['priority']}:** {rec['message']}")
                else:
                    st.success(f"{rec['icon']} {rec['message']}")
        else:
            st.success("✅ Tidak ada rekomendasi - Kepatuhan sempurna!")
        
        # Additional safety tips
        st.markdown("---")
        st.markdown("**📋 Tips Keselamatan Umum:**")
        st.markdown("""
        - Selalu lakukan inspeksi APD sebelum digunakan
        - Ganti APD yang rusak atau sudah melewati masa pakai
        - Pastikan APD sesuai dengan jenis pekerjaan yang dilakukan
        - Lakukan training rutin tentang penggunaan APD yang benar
        - Dokumentasikan setiap pelanggaran untuk evaluasi
        """)
    
    with tab4:
        st.subheader("📄 Generate Laporan PDF")
        
        st.markdown(f"""
        **Informasi Laporan:**
        - Waktu: {safety_result['timestamp']}
        - Skor Kepatuhan: {safety_result['score']:.1f}%
        - Status: {safety_result['status']}
        - Format: PDF (Portable Document Format)
        """)
        
        st.info("💡 Laporan PDF akan berisi:\n- 📸 Gambar hasil deteksi dengan bounding boxes\n- 📊 Ringkuman lengkap seperti tab 'Ringkasan'\n- 🎨 Format profesional siap cetak")
        
        # Generate PDF report with annotated image
        pdf_bytes, report_filename, saved_path = save_compliance_report(
            safety_result, 
            "uploaded_image",
            annotated_image  # Pass annotated image to include in PDF
        )
        
        if pdf_bytes:
            # Download button
            st.download_button(
                label="📥 Download Laporan PDF",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                help="Klik untuk download laporan dalam format PDF"
            )
            
            # Show save status if saved to disk
            if saved_path:
                st.success(f"✅ Laporan juga tersimpan di: `{saved_path}`")
            
            # Additional info
            st.markdown("""
            **📋 Isi Laporan PDF:**
            - ✅ Gambar hasil deteksi dengan bounding boxes APD
            - ✅ Skor kepatuhan dengan warna indikator
            - ✅ Tabel ringkasan APD terdeteksi vs kurang
            - ✅ Detail breakdown APD dengan deskripsi
            - ✅ Rekomendasi keselamatan prioritas
            - ✅ Metadata lengkap (tanggal, gambar, timestamp)
            
            **💾 Format file:** PDF dapat dibuka dengan Adobe Reader, browser, atau PDF viewer lainnya
            """)
        else:
            st.error("""
            ❌ Tidak dapat generate PDF. 
            
            **Solusi:**
            ```bash
            poetry install --extras "pdf"
            ```
            
            Atau install manual:
            ```bash
            pip install fpdf2
            ```
            """)

def display_equipment_analysis(classcounts: dict):
    """Display equipment presence analysis"""
    equipment_analysis = analyze_equipment_presence(classcounts)
    
    st.markdown("---")
    st.subheader("🏗️ Analisis Area Kerja")
    
    # Display workers count prominently
    if equipment_analysis['workers']:
        st.markdown("**👷 Pekerja Terdeteksi:**")
        for item in equipment_analysis['workers']:
            st.success(f"• {item['name'].title()}: {item['count']} orang")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if equipment_analysis['heavy_equipment']:
            st.markdown("**🚜 Alat Berat:**")
            for item in equipment_analysis['heavy_equipment']:
                st.info(f"• {item['name']}: {item['count']} unit")
        
        if equipment_analysis['tools']:
            st.markdown("**🔧 Perkakas:**")
            for item in equipment_analysis['tools']:
                st.info(f"• {item['name']}: {item['count']} unit")
    
    with col2:
        if equipment_analysis['safety_barriers']:
            st.markdown("**🚧 Safety Barriers:**")
            for item in equipment_analysis['safety_barriers']:
                st.info(f"• {item['name']}: {item['count']} unit")
        
        if equipment_analysis['other']:
            st.markdown("**📦 Lainnya:**")
            for item in equipment_analysis['other']:
                st.info(f"• {item['name']}: {item['count']} unit")

# ==================== MAIN STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Safety Detection Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🛡️ Construction Safety Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**Sistem Deteksi Kepatuhan APD: Helm & Rompi Safety**")
    st.caption("Deteksi otomatis penggunaan Helm Keselamatan dan Rompi Safety menggunakan AI")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Model selection
        selected_model = st.selectbox(
            "Pilih Model Deteksi:",
            ("Construction Equipment", "Vehicle", "Fruit"),
            help="Pilih model sesuai kebutuhan deteksi"
        )
        
        model_map = {
            "Construction Equipment": "construction",
            "Vehicle": "vehicle",
            "Fruit": "fruit"
        }
        
        st.markdown("---")
        
        # Display safety requirements
        st.subheader("📋 Standar APD Wajib")
        st.markdown("""
        Setiap pekerja **WAJIB** menggunakan:
        """)
        
        for ppe_key, ppe_info in PPE_REQUIREMENTS.items():
            st.markdown(f"🔴 {ppe_info['icon']} **{ppe_info['name_id']}**")
        
        st.markdown("---")
        st.markdown(f"""
        **ℹ️ Cara Penggunaan:**
        1. Upload foto area konstruksi
        2. Klik tombol 'Detect Objects'
        3. Review hasil deteksi & skor
        4. Download laporan jika perlu
        
        **📊 Scoring:**
        - Helmet + Vest = 100% ✅
        - Salah satu saja = 50% ⚠️
        - Tidak ada = 0% 🚨
        """)
    
    # Load model
    with st.spinner(f"Loading {selected_model} model..."):
        model = load_model(model_map[selected_model])
    
    if model is None:
        st.error("❌ Gagal memuat model. Periksa path model Anda.")
        st.stop()
    
    st.success(f"✅ {selected_model} model loaded successfully!")
    
    # File upload section
    st.markdown("---")
    st.subheader("📤 Upload Gambar untuk Deteksi")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar area konstruksi",
        accept_multiple_files=False,
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload foto dengan format JPG, JPEG, PNG, atau WEBP"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📷 Gambar Original:**")
            st.image(uploaded_file, use_container_width=True)
        
        # Detect button
        if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
            bytes_data = uploaded_file.getvalue()
            
            # Run detection with progress
            with st.spinner("🔄 Mendeteksi objek dan menganalisis keselamatan..."):
                annotated_image_rgb, classcounts, detections = detector_pipeline_pillow(bytes_data, model)
            
            if annotated_image_rgb is not None:
                # Display annotated image
                with col2:
                    st.markdown("**🎯 Hasil Deteksi:**")
                    st.image(annotated_image_rgb, use_container_width=True)
                
                # Show basic detection counts
                st.markdown("---")
                st.subheader("📊 Object Detection Results")
                
                if classcounts:
                    # Display in columns
                    cols = st.columns(min(len(classcounts), 4))
                    for idx, (class_name, count) in enumerate(classcounts.items()):
                        with cols[idx % 4]:
                            st.metric(label=class_name.replace('_', ' ').title(), value=count)
                    
                    # SAFETY COMPLIANCE DASHBOARD
                    safety_result = calculate_safety_score(classcounts)
                    display_safety_dashboard(safety_result, annotated_image_rgb)
                    
                    # EQUIPMENT ANALYSIS
                    display_equipment_analysis(classcounts)
                    
                else:
                    st.warning("⚠️ Tidak ada objek terdeteksi dalam gambar")
            else:
                st.error("❌ Gagal melakukan deteksi. Silakan coba lagi.")
    
    else:
        st.info("👆 Upload gambar untuk memulai deteksi keselamatan konstruksi")
        
        # Show sample info
        st.markdown("---")
        st.markdown("**💡 Tips untuk hasil terbaik:**")
        st.markdown("""
        - Gunakan gambar dengan pencahayaan yang baik
        - Pastikan objek APD terlihat jelas
        - Hindari gambar yang terlalu blur atau gelap
        - Foto dari jarak yang memungkinkan APD teridentifikasi
        """)

if __name__ == "__main__":
    main()
