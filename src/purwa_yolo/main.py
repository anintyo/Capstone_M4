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
    from fpdf.enums import XPos, YPos
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

# ==================== WORKER ANALYSIS CONFIGURATION ====================
# Flexible class name mapping — sesuaikan dengan nama class model YOLO Anda
WORKER_CLASS_ALIASES = [
    'person', 'Person', 'worker', 'Worker', 'human', 'Human'
]

PPE_CLASS_ALIASES = {
    'helmet': ['helmet', 'Helmet', 'Hardhat', 'Hard Hat', 'hardhat', 'hard hat'],
    'vest': ['vest', 'Vest', 'Safety Vest', 'safety vest', 'safety-vest'],
}

PPE_VIOLATION_CLASS_ALIASES = {
    'no_helmet': ['no-helmet', 'No Helmet', 'NO-Hardhat', 'no_hardhat', 'No Hardhat'],
    'no_vest': ['no-vest', 'No Vest', 'NO-Safety Vest', 'no_vest', 'No Safety Vest'],
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

# ==================== CLASS MATCHING HELPERS ====================

def match_class_count(classcounts: dict, aliases: list) -> int:
    """Sum counts for all matching aliases in classcounts."""
    total = 0
    for alias in aliases:
        total += classcounts.get(alias, 0)
    return total

def get_worker_count(classcounts: dict) -> int:
    return match_class_count(classcounts, WORKER_CLASS_ALIASES)

def get_ppe_count(classcounts: dict, ppe_key: str) -> int:
    return match_class_count(classcounts, PPE_CLASS_ALIASES.get(ppe_key, []))

def get_violation_count(classcounts: dict, violation_key: str) -> int:
    return match_class_count(classcounts, PPE_VIOLATION_CLASS_ALIASES.get(violation_key, []))

# ==================== WORKER-LEVEL SAFETY ANALYSIS ====================

def analyze_worker_compliance(classcounts: dict) -> dict:
    """
    Analisis kepatuhan APD per pekerja.

    Logika:
    1. Hitung total pekerja (kelas 'person'/'worker')
    2. Hitung APD yang terdeteksi (helmet, vest) dan pelanggaran langsung (no-helmet, no-vest)
    3. Estimasi pekerja yang tidak patuh berdasarkan:
       - Deteksi kelas pelanggaran (NO-Hardhat, NO-Safety Vest) — lebih akurat
       - Atau selisih: jumlah pekerja - jumlah APD terdeteksi
    4. Hitung compliance rate per-worker

    Returns:
        dict: detail analisis per pekerja
    """
    # Hitung total pekerja
    total_workers = get_worker_count(classcounts)

    # Hitung APD terdeteksi
    helmet_count = get_ppe_count(classcounts, 'helmet')
    vest_count = get_ppe_count(classcounts, 'vest')

    # Hitung pelanggaran yang terdeteksi langsung oleh model
    no_helmet_count = get_violation_count(classcounts, 'no_helmet')
    no_vest_count = get_violation_count(classcounts, 'no_vest')

    # ---------- Estimasi pekerja tanpa APD ----------
    # Jika model mendeteksi kelas pelanggaran secara langsung → gunakan itu
    # Jika tidak → estimasi dari selisih jumlah pekerja vs APD terdeteksi
    if total_workers > 0:
        if no_helmet_count > 0 or no_vest_count > 0:
            # Model punya kelas violation langsung (lebih akurat)
            workers_without_helmet = no_helmet_count
            workers_without_vest = no_vest_count
            workers_with_helmet = max(0, total_workers - workers_without_helmet)
            workers_with_vest = max(0, total_workers - workers_without_vest)
        else:
            # Estimasi dari count: pekerja dengan APD = min(APD count, worker count)
            workers_with_helmet = min(helmet_count, total_workers)
            workers_with_vest = min(vest_count, total_workers)
            workers_without_helmet = total_workers - workers_with_helmet
            workers_without_vest = total_workers - workers_with_vest
    else:
        # Tidak ada pekerja terdeteksi — gunakan APD count saja
        workers_with_helmet = helmet_count
        workers_with_vest = vest_count
        workers_without_helmet = no_helmet_count
        workers_without_vest = no_vest_count

    # Pekerja dengan APD LENGKAP (helmet DAN vest)
    if total_workers > 0:
        workers_fully_compliant = max(0, total_workers 
                                      - max(workers_without_helmet, workers_without_vest))
        workers_non_compliant = total_workers - workers_fully_compliant
    else:
        # Tidak bisa menentukan compliance per orang tanpa deteksi person
        workers_fully_compliant = 0
        workers_non_compliant = max(workers_without_helmet, workers_without_vest)

    return {
        'total_workers': total_workers,
        'helmet_count': helmet_count,
        'vest_count': vest_count,
        'no_helmet_count': no_helmet_count,
        'no_vest_count': no_vest_count,
        'workers_with_helmet': workers_with_helmet,
        'workers_with_vest': workers_with_vest,
        'workers_without_helmet': workers_without_helmet,
        'workers_without_vest': workers_without_vest,
        'workers_fully_compliant': workers_fully_compliant,
        'workers_non_compliant': workers_non_compliant,
        'worker_detection_available': total_workers > 0,
    }


def calculate_safety_score(classcounts: dict) -> dict:
    """
    Hitung skor kepatuhan keselamatan berbasis per-pekerja.

    Jika person terdeteksi:
        Score = (workers_fully_compliant / total_workers) * 100
    Fallback (tanpa person class):
        Score = (APD jenis terdeteksi / total APD wajib) * 100
    """
    # Analisis per-worker
    worker_analysis = analyze_worker_compliance(classcounts)

    # Tentukan skor
    if worker_analysis['worker_detection_available'] and worker_analysis['total_workers'] > 0:
        total_workers = worker_analysis['total_workers']
        fully_compliant = worker_analysis['workers_fully_compliant']
        compliance_percentage = (fully_compliant / total_workers) * 100
        score_basis = 'per_worker'
    else:
        # Fallback: cek apakah jenis APD wajib terdeteksi
        detected_classes = set(classcounts.keys())
        all_ppe_aliases = set()
        for aliases in PPE_CLASS_ALIASES.values():
            all_ppe_aliases.update(aliases)
        detected_ppe_types = detected_classes & all_ppe_aliases
        compliance_percentage = (len(detected_ppe_types) / len(PPE_CLASS_ALIASES)) * 100 if PPE_CLASS_ALIASES else 0
        score_basis = 'ppe_type'

    # Tentukan status berdasarkan skor
    if compliance_percentage >= SCORE_THRESHOLDS['excellent']:
        status, status_color, status_icon = 'SANGAT BAIK', 'green', '✅'
    elif compliance_percentage >= SCORE_THRESHOLDS['good']:
        status, status_color, status_icon = 'BAIK', 'blue', '👍'
    elif compliance_percentage >= SCORE_THRESHOLDS['fair']:
        status, status_color, status_icon = 'CUKUP', 'orange', '⚠️'
    else:
        status, status_color, status_icon = 'KURANG', 'red', '🚨'

    # Deteksi APD jenis untuk rekomendasi
    detected_ppe_keys = []
    missing_ppe_keys = []
    for ppe_key, aliases in PPE_CLASS_ALIASES.items():
        if any(classcounts.get(a, 0) > 0 for a in aliases):
            detected_ppe_keys.append(ppe_key)
        else:
            # Cek juga dari worker analysis
            if ppe_key == 'helmet' and worker_analysis['workers_with_helmet'] > 0:
                detected_ppe_keys.append(ppe_key)
            elif ppe_key == 'vest' and worker_analysis['workers_with_vest'] > 0:
                detected_ppe_keys.append(ppe_key)
            else:
                missing_ppe_keys.append(ppe_key)

    recommendations = generate_safety_recommendations(missing_ppe_keys, worker_analysis)

    return {
        'score': compliance_percentage,
        'score_basis': score_basis,
        'status': status,
        'status_color': status_color,
        'status_icon': status_icon,
        'detected_ppe': detected_ppe_keys,
        'missing_ppe': missing_ppe_keys,
        'detected_counts': classcounts,
        'worker_analysis': worker_analysis,
        'recommendations': recommendations,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def generate_safety_recommendations(missing_ppe: list, worker_analysis: dict) -> list:
    """Generate actionable safety recommendations."""
    recommendations = []

    wa = worker_analysis

    # Rekomendasi berbasis pekerja tanpa APD
    if wa['workers_without_helmet'] > 0:
        recommendations.append({
            'priority': 'CRITICAL',
            'message': f"SEGERA: {wa['workers_without_helmet']} pekerja tidak menggunakan Helm Keselamatan!",
            'icon': '🚨'
        })

    if wa['workers_without_vest'] > 0:
        recommendations.append({
            'priority': 'CRITICAL',
            'message': f"SEGERA: {wa['workers_without_vest']} pekerja tidak menggunakan Rompi Safety!",
            'icon': '🚨'
        })

    # Jika tidak ada deteksi pekerja tapi ada APD yang kurang
    for ppe in missing_ppe:
        item = PPE_REQUIREMENTS.get(ppe, {})
        if item and ppe not in ['helmet', 'vest']:  # hindari duplikasi
            recommendations.append({
                'priority': 'CRITICAL',
                'message': f"SEGERA pastikan semua pekerja menggunakan {item.get('name_id', ppe)}",
                'icon': '🚨'
            })

    if not recommendations:
        recommendations.append({
            'priority': 'INFO',
            'message': "Semua pekerja telah menggunakan APD dengan lengkap. Pertahankan!",
            'icon': '👍'
        })

    return recommendations

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

def generate_pdf_report(safety_result: dict, image_name: str, annotated_image: np.ndarray = None) -> bytes:
    """
    Generate PDF report with format matching Ringkuman tab
    Returns PDF as bytes for download
    """
    if not PDF_AVAILABLE:
        raise ImportError("fpdf2 not installed. Install with: poetry install --extras 'pdf'")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    
    # ============ HEADER ============
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(220, 53, 69)
    pdf.cell(0, 15, 'Safety Compliance Dashboard', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, 'Sistem Deteksi Kepatuhan APD: Helm & Rompi Safety', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # ============ METADATA ============
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 5, f'Tanggal: {safety_result["timestamp"]}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, f'Gambar: {image_name}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # ============ ANNOTATED IMAGE ============
    if annotated_image is not None:
        try:
            import tempfile, os
            if annotated_image.dtype != np.uint8:
                annotated_image = annotated_image.astype(np.uint8)
            pil_img = Image.fromarray(annotated_image)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_path = tmp_file.name
                pil_img.save(temp_path, 'JPEG', quality=85)
            
            max_width = 180
            img_width, img_height = pil_img.width, pil_img.height
            if img_width > img_height:
                pdf_img_width = max_width
                pdf_img_height = (img_height / img_width) * max_width
            else:
                max_height = 100
                scale = min(max_width / img_width, max_height / img_height)
                pdf_img_width = img_width * scale
                pdf_img_height = img_height * scale
            
            x = (pdf.w - pdf_img_width) / 2
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 6, 'Gambar Hasil Deteksi:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            pdf.image(temp_path, x=x, y=pdf.get_y(), w=pdf_img_width, h=pdf_img_height)
            pdf.ln(pdf_img_height + 5)
            
            try:
                os.unlink(temp_path)
            except:
                pass
        except Exception as e:
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 5, f'[Gambar tidak dapat ditampilkan: {str(e)}]', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
    
    # ============ SCORE BOX ============
    score = safety_result['score']
    if score >= 90:
        bg_color = (40, 167, 69); status_symbol = '[EXCELLENT]'
    elif score >= 75:
        bg_color = (0, 123, 255); status_symbol = '[GOOD]'
    elif score >= 60:
        bg_color = (255, 193, 7); status_symbol = '[FAIR]'
    else:
        bg_color = (220, 53, 69); status_symbol = '[POOR]'
    
    pdf.set_fill_color(*bg_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 32)
    box_width = 180
    x = (pdf.w - box_width) / 2
    pdf.set_xy(x, pdf.get_y())
    pdf.cell(box_width, 30, '', align='C', fill=True)
    pdf.set_xy(x, pdf.get_y() + 5)
    pdf.cell(box_width, 10, f'{score:.1f}%', align='C')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_xy(x, pdf.get_y() + 12)
    pdf.cell(box_width, 10, safety_result['status'], align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # ============ WORKER ANALYSIS SUMMARY ============
    wa = safety_result.get('worker_analysis', {})
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Ringkasan Analisis Pekerja', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    
    # Worker stats table
    col_width = 57
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(200, 200, 200)
    for header in ['Total Pekerja', 'Helm Dipakai', 'Rompi Dipakai']:
        pdf.cell(col_width, 8, header, border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_fill_color(240, 248, 255)
    total_workers = wa.get('total_workers', '-')
    workers_with_helmet = wa.get('workers_with_helmet', wa.get('helmet_count', '-'))
    workers_with_vest = wa.get('workers_with_vest', wa.get('vest_count', '-'))
    for val in [str(total_workers), str(workers_with_helmet), str(workers_with_vest)]:
        pdf.cell(col_width, 8, val, border=1, align='C', fill=True)
    pdf.ln(12)
    
    # Non-compliance stats
    col_width = 90
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(200, 200, 200)
    for header in ['Tanpa Helm', 'Tanpa Rompi']:
        pdf.cell(col_width, 8, header, border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 10)
    no_helmet = wa.get('workers_without_helmet', '-')
    no_vest = wa.get('workers_without_vest', '-')
    
    for val, color in [(str(no_helmet), (255, 220, 220)), (str(no_vest), (255, 220, 220))]:
        pdf.set_fill_color(*color)
        pdf.cell(col_width, 8, val, border=1, align='C', fill=True)
    pdf.ln(12)
    
    # ============ RINGKASAN KEPATUHAN APD ============
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Ringkasan Kepatuhan APD', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    
    col_widths = [30, 80, 40, 30]
    headers = ['Status', 'APD', 'Prioritas', 'Jumlah']
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(200, 200, 200)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_fill_color(240, 255, 240)
    for ppe in safety_result['detected_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        pdf.set_text_color(0, 128, 0)
        pdf.cell(col_widths[0], 7, 'Terdeteksi', border=1, align='C', fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[1], 7, ppe_info.get('name_id', ppe), border=1, align='L', fill=True)
        pdf.set_text_color(220, 53, 69)
        pdf.cell(col_widths[2], 7, ppe_info.get('priority', 'CRITICAL'), border=1, align='C', fill=True)
        pdf.set_text_color(0, 0, 0)
        count = safety_result['detected_counts'].get(ppe, 0)
        pdf.cell(col_widths[3], 7, str(count), border=1, align='C', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_fill_color(255, 240, 240)
    for ppe in safety_result['missing_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        pdf.set_text_color(220, 53, 69)
        pdf.cell(col_widths[0], 7, 'Kurang', border=1, align='C', fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[1], 7, ppe_info.get('name_id', ppe), border=1, align='L', fill=True)
        pdf.set_text_color(220, 53, 69)
        pdf.cell(col_widths[2], 7, 'CRITICAL', border=1, align='C', fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(col_widths[3], 7, '0', border=1, align='C', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # ============ REKOMENDASI ============
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Rekomendasi Keselamatan', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 9)
    
    for i, rec in enumerate(safety_result['recommendations'], 1):
        if rec['priority'] == 'CRITICAL':
            pdf.set_text_color(220, 53, 69)
            priority_label = '[CRITICAL] '
        else:
            pdf.set_text_color(0, 0, 0)
            priority_label = ''
        pdf.cell(10)
        pdf.multi_cell(0, 5, f"{i}. {priority_label}{rec['message']}")
        pdf.ln(1)
    
    pdf.ln(10)
    
    # ============ FOOTER ============
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, 'Laporan dibuat secara otomatis oleh Safety Compliance Dashboard', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, 'Sistem Deteksi Keselamatan Konstruksi berbasis AI', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    return bytes(pdf.output())


def save_compliance_report(safety_result: dict, image_name: str, annotated_image: np.ndarray = None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"safety_report_{timestamp}.pdf"
    
    try:
        pdf_bytes = generate_pdf_report(safety_result, image_name, annotated_image)
        try:
            report_path = REPORTS_DIR / report_filename
            with open(report_path, 'wb') as f:
                f.write(pdf_bytes)
            saved_path = str(report_path)
        except Exception:
            saved_path = None
        return pdf_bytes, report_filename, saved_path
    except ImportError:
        st.error("⚠️ PDF library tidak terinstall. Install dengan: poetry install --extras 'pdf'")
        return None, None, None


def generate_report_summary(safety_result: dict) -> pd.DataFrame:
    """Generate summary table for display (used in Ringkasan tab)"""
    data = []
    for ppe in safety_result['detected_ppe']:
        ppe_info = PPE_REQUIREMENTS.get(ppe, {})
        data.append({
            'Status': '✅ Terdeteksi',
            'APD': ppe_info.get('name_id', ppe),
            'Prioritas': ppe_info.get('priority', 'N/A'),
            'Jumlah': safety_result['detected_counts'].get(ppe, 0)
        })
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

def display_worker_analysis_banner(worker_analysis: dict):
    """
    Tampilkan banner analisis per-pekerja sesuai brief:
    Contoh: '4 pekerja: 4 helmet, 2 vest, 2 no-vest → 2 orang tidak lengkap'
    """
    wa = worker_analysis
    st.markdown("---")
    st.subheader("👷 Analisis Kepatuhan Per Pekerja")

    if wa['worker_detection_available']:
        total = wa['total_workers']
        # Narasi seperti brief
        st.info(
            f"📌 Terdeteksi **{total} pekerja** di gambar:\n\n"
            f"- 🪖 Dengan Helm: **{wa['workers_with_helmet']}** orang\n"
            f"- 🦺 Dengan Rompi: **{wa['workers_with_vest']}** orang\n"
            f"- ⛑ Tanpa Helm: **{wa['workers_without_helmet']}** orang\n"
            f"- 🚫 Tanpa Rompi: **{wa['workers_without_vest']}** orang\n\n"
            f"👉 **{wa['workers_non_compliant']} pekerja tidak memakai APD secara lengkap**"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pekerja", total)
        col2.metric("Helm ✅", wa['workers_with_helmet'],
                    delta=f"-{wa['workers_without_helmet']} tanpa helm" if wa['workers_without_helmet'] else None,
                    delta_color="inverse")
        col3.metric("Rompi ✅", wa['workers_with_vest'],
                    delta=f"-{wa['workers_without_vest']} tanpa rompi" if wa['workers_without_vest'] else None,
                    delta_color="inverse")
        col4.metric("Tidak Patuh 🚨", wa['workers_non_compliant'])
    else:
        st.warning(
            "⚠️ Kelas 'person/worker' tidak terdeteksi dalam gambar ini.\n\n"
            "Analisis per-pekerja tidak tersedia. Pastikan model Anda dapat mendeteksi kelas pekerja, "
            "atau lihat ringkasan APD di bawah."
        )
        col1, col2 = st.columns(2)
        col1.metric("Helm Terdeteksi 🪖", wa['helmet_count'])
        col2.metric("Rompi Terdeteksi 🦺", wa['vest_count'])
        if wa['no_helmet_count'] > 0 or wa['no_vest_count'] > 0:
            col3, col4 = st.columns(2)
            col3.metric("Tanpa Helm ⚠️", wa['no_helmet_count'])
            col4.metric("Tanpa Rompi ⚠️", wa['no_vest_count'])


def display_safety_dashboard(safety_result: dict, annotated_image: np.ndarray = None):
    """Display comprehensive safety compliance dashboard"""
    
    st.markdown("---")
    st.header("🛡️ Safety Compliance Dashboard")
    
    # Score Display
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        score_label = "Per-Worker Compliance" if safety_result['score_basis'] == 'per_worker' else "APD Type Coverage"
        st.markdown(f"""
        <div style='padding: 20px; background-color: {safety_result['status_color']}; 
        border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>{safety_result['status_icon']} {safety_result['score']:.1f}%</h1>
            <h3 style='color: white; margin: 0;'>{safety_result['status']}</h3>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85em;'>{score_label}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric("APD Terdeteksi", len(safety_result['detected_ppe']))
    with col3:
        st.metric("APD Kurang", len(safety_result['missing_ppe']))
    
    # Worker Analysis Banner (NEW — sesuai brief)
    display_worker_analysis_banner(safety_result['worker_analysis'])
    
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
                    st.error(ppe_info['priority'])
    
    with tab3:
        st.subheader("💡 Rekomendasi Keselamatan")
        
        if safety_result['recommendations']:
            for i, rec in enumerate(safety_result['recommendations'], 1):
                if rec['priority'] == 'CRITICAL':
                    st.error(f"{rec['icon']} **{rec['priority']}:** {rec['message']}")
                else:
                    st.success(f"{rec['icon']} {rec['message']}")
        else:
            st.success("✅ Tidak ada rekomendasi - Kepatuhan sempurna!")
        
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
        
        st.info("💡 Laporan PDF akan berisi:\n- 📸 Gambar hasil deteksi dengan bounding boxes\n"
                "- 👷 Ringkasan analisis per-pekerja\n- 📊 Ringkuman lengkap APD\n"
                "- 🎨 Format profesional siap cetak")
        
        pdf_bytes, report_filename, saved_path = save_compliance_report(
            safety_result, 
            "uploaded_image",
            annotated_image
        )
        
        if pdf_bytes:
            st.download_button(
                label="📥 Download Laporan PDF",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                help="Klik untuk download laporan dalam format PDF"
            )
            if saved_path:
                st.success(f"✅ Laporan juga tersimpan di: `{saved_path}`")
        else:
            st.error("""
            ❌ Tidak dapat generate PDF. 
            
            **Solusi:**
            ```bash
            poetry install --extras "pdf"
            ```
            """)

# ==================== MAIN STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Safety Detection Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
    st.caption("Deteksi otomatis penggunaan Helm Keselamatan dan Rompi Safety menggunakan AI (YOLOv8)")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
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
        
        st.subheader("📋 Standar APD Wajib")
        st.markdown("Setiap pekerja **WAJIB** menggunakan:")
        for ppe_key, ppe_info in PPE_REQUIREMENTS.items():
            st.markdown(f"🔴 {ppe_info['icon']} **{ppe_info['name_id']}**")
        
        st.markdown("---")
        st.markdown(f"""
        **ℹ️ Cara Penggunaan:**
        1. Upload foto area konstruksi
        2. Klik tombol 'Detect Objects'
        3. Review hasil deteksi & skor per pekerja
        4. Download laporan jika perlu
        
        **📊 Scoring (per-pekerja):**
        - Semua pekerja lengkap = 100% ✅
        - Sebagian pekerja lengkap = 50-99% ⚠️
        - Tidak ada yang lengkap = 0% 🚨
        """)
    
    # Load model
    with st.spinner(f"Loading {selected_model} model..."):
        model = load_model(model_map[selected_model])
    
    if model is None:
        st.error("❌ Gagal memuat model. Periksa path model Anda.")
        st.stop()
    
    st.success(f"✅ {selected_model} model loaded successfully!")
    
    # File upload
    st.markdown("---")
    st.subheader("📤 Upload Gambar untuk Deteksi")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar area konstruksi",
        accept_multiple_files=False,
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload foto dengan format JPG, JPEG, PNG, atau WEBP"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**📷 Gambar Original:**")
            st.image(uploaded_file, use_container_width=True)
        
        if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
            bytes_data = uploaded_file.getvalue()
            
            with st.spinner("🔄 Mendeteksi objek dan menganalisis keselamatan..."):
                annotated_image_rgb, classcounts, detections = detector_pipeline_pillow(bytes_data, model)
            
            if annotated_image_rgb is not None:
                with col2:
                    st.markdown("**🎯 Hasil Deteksi:**")
                    st.image(annotated_image_rgb, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📊 Object Detection Results")
                
                if classcounts:
                    cols = st.columns(min(len(classcounts), 4))
                    for idx, (class_name, count) in enumerate(classcounts.items()):
                        with cols[idx % 4]:
                            st.metric(label=class_name.replace('_', ' ').title(), value=count)
                    
                    # Full Safety Compliance Dashboard (dengan analisis per-pekerja)
                    safety_result = calculate_safety_score(classcounts)
                    display_safety_dashboard(safety_result, annotated_image_rgb)
                else:
                    st.warning("⚠️ Tidak ada objek terdeteksi dalam gambar")
            else:
                st.error("❌ Gagal melakukan deteksi. Silakan coba lagi.")
    
    else:
        st.info("👆 Upload gambar untuk memulai deteksi keselamatan konstruksi")
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
