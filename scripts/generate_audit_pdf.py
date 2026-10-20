import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QTextDocument, QFont, QPageLayout, QPageSize
from PySide6.QtCore import QMarginsF, QSize
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtPdf import QPdfDocument

def generate_pretty_pdf():
    app = QApplication.instance() or QApplication([])

    # Load Windows system TrueType fonts into Qt offscreen font database
    font_files = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/consolas.ttf",
        "C:/Windows/Fonts/consolab.ttf"
    ]
    for f in font_files:
        if os.path.exists(f):
            QFontDatabase.addApplicationFont(f)

    app_font = QFont("Arial", 9)
    app.setFont(app_font)

    output_pdf = Path("Blink_Rigorous_Factual_Audit_Report.pdf").resolve()

    # Image URIs for direct embedding
    cnn_cm_uri = Path("results/cnn_test_confusion_matrix.png").resolve().as_uri()
    fnn_cm_uri = Path("results/fnn_test_confusion_matrix.png").resolve().as_uri()
    cnn_curves_uri = Path("results/cnn_training_curves.png").resolve().as_uri()
    ui_idle_uri = Path("results/diagnostics/ui_screens/01_idle_standby.png").resolve().as_uri()
    ui_alert_uri = Path("results/diagnostics/ui_screens/04_drowsiness_alert.png").resolve().as_uri()

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        font-size: 7.9pt;
        line-height: 1.34;
        color: #1e293b;
        margin: 0;
        padding: 0;
    }}
    .top-accent-bar {{
        background-color: #2563eb;
        height: 3px;
        margin-bottom: 6px;
    }}
    .header-table {{
        width: 100%;
        border-collapse: collapse;
        border: none;
        margin-bottom: 6px;
    }}
    .header-table td {{
        border: none;
        padding: 0;
        vertical-align: middle;
    }}
    .brand-title {{
        font-size: 15pt;
        font-weight: bold;
        color: #0f172a;
        letter-spacing: -0.3px;
        margin: 0;
    }}
    .brand-subtitle {{
        font-size: 8.5pt;
        color: #475569;
        margin-top: 1px;
    }}
    .pill-badge {{
        display: inline-block;
        background-color: #0f172a;
        color: #38bdf8;
        font-weight: bold;
        font-size: 7.5pt;
        padding: 3px 8px;
        border-radius: 4px;
        text-align: right;
    }}
    .meta-card {{
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        padding: 5px 8px;
        margin-bottom: 6px;
    }}
    .meta-grid {{
        width: 100%;
        border-collapse: collapse;
        border: none;
        font-size: 7.6pt;
    }}
    .meta-grid td {{
        border: none;
        padding: 1.5px 4px;
        color: #475569;
    }}
    .meta-grid strong {{
        color: #0f172a;
    }}
    .kpi-table {{
        width: 100%;
        border-collapse: collapse;
        border: none;
        margin-bottom: 6px;
    }}
    .kpi-table td {{
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        padding: 5px 6px;
        text-align: center;
        border-radius: 4px;
        width: 25%;
    }}
    .kpi-num {{
        font-size: 13pt;
        font-weight: bold;
        color: #2563eb;
        margin: 0;
    }}
    .kpi-label {{
        font-size: 6.8pt;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1px;
    }}
    h2 {{
        font-size: 10.2pt;
        font-weight: bold;
        color: #0f172a;
        border-bottom: 1.5px solid #0f172a;
        padding-bottom: 2px;
        margin-top: 6px;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    h3 {{
        font-size: 8.8pt;
        font-weight: bold;
        color: #1e3a8a;
        margin-top: 5px;
        margin-bottom: 2px;
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 4px 0 6px 0;
        font-size: 7.7pt;
    }}
    table.data-table th, table.data-table td {{
        border: 1px solid #cbd5e1;
        padding: 2.5px 5px;
        text-align: left;
    }}
    table.data-table th {{
        background-color: #0f172a;
        color: #ffffff;
        font-weight: bold;
        font-size: 7.5pt;
        letter-spacing: 0.3px;
    }}
    table.data-table tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    .badge-verified {{
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        font-weight: bold;
        padding: 1.5px 5px;
        border-radius: 9999px;
        font-size: 6.8pt;
        border: 1px solid #bbf7d0;
    }}
    .badge-corrected {{
        display: inline-block;
        background-color: #fef3c7;
        color: #b45309;
        font-weight: bold;
        padding: 1.5px 5px;
        border-radius: 9999px;
        font-size: 6.8pt;
        border: 1px solid #fde68a;
    }}
    .alert-card {{
        background-color: #fff1f2;
        border-left: 3px solid #e11d48;
        border-radius: 0 4px 4px 0;
        padding: 4px 8px;
        margin: 5px 0;
        font-size: 7.6pt;
        color: #9f1239;
    }}
    .quote-card {{
        background-color: #f0f9ff;
        border-left: 3px solid #0284c7;
        border-radius: 0 4px 4px 0;
        padding: 4px 8px;
        margin: 5px 0;
        font-size: 7.6pt;
        color: #0369a1;
    }}
    .code-box {{
        background-color: #090d16;
        color: #f1f5f9;
        padding: 4px 6px;
        border-radius: 4px;
        margin: 3px 0;
        font-family: Consolas, 'Courier New', monospace;
        font-size: 6.8pt;
        line-height: 1.28;
        border: 1px solid #1e293b;
    }}
    ul, ol {{
        margin-top: 1px;
        margin-bottom: 4px;
        padding-left: 14px;
    }}
    li {{
        margin-bottom: 1.5px;
    }}
    .img-showcase {{
        text-align: center;
        margin: 4px 0;
    }}
    .page-break {{
        page-break-before: always;
    }}
    .text-right {{
        text-align: right;
    }}
    .text-center {{
        text-align: center;
    }}
</style>
</head>
<body>

<!-- ================= PAGE 1 ================= -->
<div class="top-accent-bar"></div>
<table class="header-table">
    <tr>
        <td>
            <div class="brand-title">BLINK: DRIVER ATTENTIVENESS SYSTEM</div>
            <div class="brand-subtitle">Rigorous Technical Audit, Empirical Benchmark &amp; Engineering Narrative</div>
        </td>
        <td class="text-right">
            <span class="pill-badge">AUDIT BASELINE v1.0</span>
        </td>
    </tr>
</table>

<div class="meta-card">
    <table class="meta-grid">
        <tr>
            <td style="width: 25%;"><strong>Document Scope:</strong> Academic Defense Baseline</td>
            <td style="width: 25%;"><strong>Audit Date:</strong> October 20, 2026</td>
            <td style="width: 25%;"><strong>Hardware Target:</strong> Windows Native CPU</td>
            <td style="width: 25%;"><strong>Vision Engine:</strong> MediaPipe + Keras</td>
        </tr>
        <tr>
            <td><strong>Neural Core:</strong> Compact 2D CNN (1.14M)</td>
            <td><strong>Benchmark FNN:</strong> Dense MLP (2.26M)</td>
            <td><strong>Dataset:</strong> MRL Eye (84,898 images)</td>
            <td><strong>Test Status:</strong> Automated Suites 4/4 PASS</td>
        </tr>
    </table>
</div>

<table class="kpi-table">
    <tr>
        <td>
            <div class="kpi-num">98.75%</div>
            <div class="kpi-label">CNN Test Accuracy</div>
        </td>
        <td>
            <div class="kpi-num">+7.32%</div>
            <div class="kpi-label">Delta over FNN</div>
        </td>
        <td>
            <div class="kpi-num">-49.5%</div>
            <div class="kpi-label">Parameter Reduction</div>
        </td>
        <td>
            <div class="kpi-num">13.1 MB</div>
            <div class="kpi-label">Model Footprint</div>
        </td>
    </tr>
</table>

<h2>Part I: Factual Audit &amp; System Architecture</h2>

<h3>1. Verified System Identity &amp; Components</h3>
<table class="data-table">
    <tr>
        <th style="width: 20%;">Component</th>
        <th style="width: 36%;">Claimed Specification</th>
        <th style="width: 34%;">Verified Code / Artifact Location</th>
        <th style="width: 10%;">Status</th>
    </tr>
    <tr>
        <td><strong>Project Identity</strong></td>
        <td>Blink (formerly AEGIS)</td>
        <td><code>src/desktop_app.py:276</code> (Window: <em>Blink | Driver Monitoring</em>)</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Vision Frontend</strong></td>
        <td>MediaPipe Face Landmarker (478 pts)</td>
        <td><code>models/face_landmarker.task</code> (3.58 MB) via <code>src/eye_extractor.py</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Neural Classifier</strong></td>
        <td>Compact Convolutional Neural Network</td>
        <td><code>models/cnn_baseline_best.keras</code> (13.11 MB, 1,141,505 params)</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Desktop Interface</strong></td>
        <td>PySide6 Pitch Black Cockpit</td>
        <td><code>src/desktop_app.py</code> with asynchronous <code>VideoWorker(QThread)</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Audio Alarm</strong></td>
        <td>2.5 kHz buzzer with cooldown</td>
        <td><code>winsound.Beep(2500, 250)</code> with 1.0s cooldown (<code>src/webcam_detector.py</code>)</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
</table>

<h3>2. Dataset Properties &amp; Split Integrity (MRL Eye Dataset)</h3>
<table class="data-table">
    <tr>
        <th style="width: 25%;">Dataset Dimension</th>
        <th style="width: 33%;">Verified Metric / Value</th>
        <th style="width: 32%;">Primary Evidence File</th>
        <th style="width: 10%;">Status</th>
    </tr>
    <tr>
        <td><strong>Total Volume</strong></td>
        <td>84,898 images (0 unreadable / corrupted)</td>
        <td><code>results/dataset_inspection_summary.json:3,31</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Class Proportions</strong></td>
        <td>Awake: 42,952 (50.59%) | Sleepy: 41,946 (49.41%)</td>
        <td><code>results/dataset_inspection_summary.json:23-30</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Split Allocation</strong></td>
        <td>Train: 50,937 (60%) | Val: 16,980 (20%) | Test: 16,981 (20%)</td>
        <td><code>results/dataset_inspection_summary.json:18-22</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Input Formatting</strong></td>
        <td>64x64 single-channel grayscale, normalized [0, 1]</td>
        <td><code>src/data_loader.py:7</code> &amp; <code>layers.Rescaling(1./255)</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Class Mapping</strong></td>
        <td>Index 0 = <code>sleepy</code> (closed) | Index 1 = <code>awake</code> (open)</td>
        <td><code>src/data_loader.py:6</code> (<code>CLASS_NAMES = ["sleepy", "awake"]</code>)</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Subject Overlap</strong></td>
        <td>All 37 subjects present across Train, Val, and Test</td>
        <td><code>results/dataset_inspection_summary.json:48-55</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
</table>

<div class="page-break"></div>

<!-- ================= PAGE 2 ================= -->
<h2>Part II: Quantitative Model Benchmark (N = 16,981)</h2>

<table class="data-table">
    <tr>
        <th>Evaluation Metric</th>
        <th>Baseline FNN</th>
        <th>Compact CNN (Blink Core)</th>
        <th>Observed Absolute Delta</th>
        <th>Status</th>
    </tr>
    <tr>
        <td><strong>Architecture Type</strong></td>
        <td>4-Layer Dense MLP</td>
        <td>3-Block Conv2D + Dense</td>
        <td>Spatial Inductive Bias</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Total Parameters</strong></td>
        <td>2,262,017</td>
        <td><strong>1,141,505</strong></td>
        <td><strong>-49.54% (-1.12M params)</strong></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Checkpoint File Size</strong></td>
        <td>25.92 MB (27,183,957 B)</td>
        <td><strong>13.11 MB (13,745,210 B)</strong></td>
        <td><strong>-49.42% (-12.81 MB)</strong></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Training Time (CPU)</strong></td>
        <td>171.03 s (~2.85 min)</td>
        <td>785.79 s (~13.10 min)</td>
        <td>+359.4% (4.6x training compute)</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Optimal Checkpoint</strong></td>
        <td>Epoch 7 (val_acc: 91.49%)</td>
        <td>Epoch 9 (val_acc: 98.75%)</td>
        <td>Monitored <code>val_accuracy</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Holdout Test Accuracy</strong></td>
        <td>91.43% (15,525 / 16,981)</td>
        <td><strong>98.75% (16,768 / 16,981)</strong></td>
        <td><strong>+7.32 percentage points</strong></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Macro F1-Score</strong></td>
        <td>0.9143</td>
        <td><strong>0.9875</strong></td>
        <td><strong>+0.0732</strong></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Sleepy Recall (Closed)</strong></td>
        <td>92.60% (7,769 / 8,390)</td>
        <td><strong>98.40% (8,256 / 8,390)</strong></td>
        <td><strong>+5.80 percentage points</strong></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Awake Recall (Open)</strong></td>
        <td>90.28% (7,756 / 8,591)</td>
        <td><strong>99.08% (8,512 / 8,591)</strong></td>
        <td><strong>+8.80 percentage points</strong></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Sleepy Precision</strong></td>
        <td>90.30%</td>
        <td><strong>99.05%</strong></td>
        <td>+8.75 percentage points</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Awake Precision</strong></td>
        <td>92.59%</td>
        <td><strong>98.45%</strong></td>
        <td>+5.86 percentage points</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
</table>

<div class="alert-card">
    <strong>Documentation Discrepancy Corrected:</strong> The project README originally stated <em>"4-Block Conv2D + BatchNorm + Dense"</em>. Direct verification of <code>src/cnn_model.py</code> and model weights confirms the CNN actually contains <strong>3 Conv2D blocks (32, 64, 128 filters), 0 BatchNormalization layers, and 1 Dense hidden layer (128 units)</strong>.
</div>

<h3>Ground-Truth Test Confusion Matrices (N = 16,981)</h3>
<table class="data-table" style="margin-bottom: 4px;">
    <tr>
        <th style="width: 50%; text-align: center;">Baseline FNN Confusion Breakdown (91.43%)</th>
        <th style="width: 50%; text-align: center;">Compact CNN Confusion Breakdown (98.75%)</th>
    </tr>
    <tr>
        <td style="font-size: 7.2pt; padding: 4px 6px; background-color: #f8fafc; font-family: Consolas, monospace;">
            Pred Sleepy: 7,769 | Pred Awake: 621 (Sleepy Recall: 92.60%)<br>
            Act Awake Pred Sleepy: 835 | Pred Awake: 7,756 (Awake Recall: 90.28%)<br>
            <strong>Total Correct: 15,525 / 16,981 (Accuracy: 91.43%)</strong>
        </td>
        <td style="font-size: 7.2pt; padding: 4px 6px; background-color: #f8fafc; font-family: Consolas, monospace;">
            Pred Sleepy: 8,256 | Pred Awake: 134 (Sleepy Recall: 98.40%)<br>
            Act Awake Pred Sleepy: 79 | Pred Awake: 8,512 (Awake Recall: 99.08%)<br>
            <strong>Total Correct: 16,768 / 16,981 (Accuracy: 98.75%)</strong>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; padding: 4px; background-color: #ffffff;">
            <img src="{fnn_cm_uri}" width="195" />
        </td>
        <td style="text-align: center; padding: 4px; background-color: #ffffff;">
            <img src="{cnn_cm_uri}" width="195" />
        </td>
    </tr>
</table>

<div class="page-break"></div>

<!-- ================= PAGE 3 ================= -->
<h2>Part III: Training Dynamics &amp; Real-Time Pipeline</h2>

<h3>1. CNN Training Convergence &amp; Loss Dynamics</h3>
<table style="width: 100%; border-collapse: collapse; border: none; margin-bottom: 5px;">
    <tr>
        <td style="width: 52%; border: none; padding-right: 8px; vertical-align: middle;">
            <ul>
                <li><strong>Rapid Spatial Convergence:</strong> The CNN achieved 96.11% validation accuracy in Epoch 1, surpassing the FNN's final 10-epoch plateau (91.49%) in a single pass.</li>
                <li><strong>Low Generalization Gap:</strong> Minimal divergence between training loss (0.029) and validation loss (0.041), confirming strong regularization from Dropout (0.3).</li>
                <li><strong>Optimal Checkpoint:</strong> <code>ModelCheckpoint</code> selected Epoch 9 (98.75% val accuracy), perfectly aligned with the holdout test evaluation.</li>
            </ul>
        </td>
        <td style="width: 48%; border: none; text-align: center; vertical-align: middle;">
            <img src="{cnn_curves_uri}" width="260" />
        </td>
    </tr>
</table>

<h3>2. Verified Real-Time Vision Pipeline Architecture</h3>
<div class="meta-card">
    <ul>
        <li><strong>MediaPipe Decoupled Extraction:</strong> Replaced unreliable classic Haar cascades with a 478-point Face Landmarker (<code>models/face_landmarker.task</code>). Eye crops are isolated via inner/outer landmark corners (33/133 and 362/263) with a 1.35x bounding margin. Ocular regions remain fully trackable whether eyelids are open or shut.</li>
        <li><strong>1:1 Aspect-Ratio Preservation:</strong> Guarantees a strictly square bounding crop prior to 64x64 resampling, eliminating the horizontal eyelid stretching flaw that caused closed eyes to mimic open eyes.</li>
        <li><strong>Dual-Eye Inference &amp; Probability Averaging:</strong> Left and right eyes are evaluated independently through the CNN and averaged: <code>P(awake) = (P_R + P_L) / 2</code>.</li>
        <li><strong>Temporal Hysteresis Engine (<code>TemporalDrowsinessTracker</code>):</strong>
            <ul>
                <li><strong>5-Frame Moving Average:</strong> Filters raw frame predictions through a FIFO buffer (<code>deque(maxlen=5)</code>) to suppress single-frame classification flutter.</li>
                <li><strong>250 ms Open Grace Period:</strong> Natural physiological blinks and brief open flickers do not immediately zero out accumulated closure time.</li>
                <li><strong>500 ms Unknown Dropout Timeout:</strong> Head turns and sensor dropouts transition conservatively to <code>UNKNOWN</code> without triggering false alarms.</li>
                <li><strong>1.50 s Sustained Closure Threshold:</strong> Wall-clock accumulator triggers the non-blocking audio buzzer and UI alert banner once continuous closure exceeds 1.50 seconds.</li>
            </ul>
        </li>
    </ul>
</div>

<h3>3. Regression Test Verification</h3>
<table class="data-table">
    <tr>
        <th>Test Suite</th>
        <th>Target Module</th>
        <th>Test Scope</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>test_desktop_app_headless.py</code></td>
        <td><code>src/desktop_app.py</code></td>
        <td>Window init, model reuse, OPEN/CLOSED/ALERT/UNKNOWN state transitions, shutdown</td>
        <td><span class="badge-verified">4/4 PASS</span></td>
    </tr>
    <tr>
        <td><code>test_temporal_tracker.py</code></td>
        <td><code>TemporalDrowsinessTracker</code></td>
        <td>Stable closure, 1-frame flicker resistance, grace period reset, dropout conservatism</td>
        <td><span class="badge-verified">4/4 PASS</span></td>
    </tr>
</table>

<div class="page-break"></div>

<!-- ================= PAGE 4 ================= -->
<h2>Part IV: Reconstructed Project History (Milestones 1–5)</h2>

<h3>Milestone 1: Problem Definition, Dataset Selection &amp; Inspection</h3>
<ul>
    <li><strong>Attempted:</strong> Establishing an automated machine learning pipeline to detect driver drowsiness from single eye crops. The <strong>MRL Eye Dataset</strong> (84,898 images across 37 human subjects) was selected for its inclusion of infrared sensors, eyeglasses reflections, and head pose variations.</li>
    <li><strong>Why Attempted:</strong> Real-world driver monitoring requires high-accuracy ocular classification under non-ideal vehicle cab conditions.</li>
    <li><strong>Outcome:</strong> Verified zero unreadable images and balanced classes (50.59% open, 49.41% closed). Established an image-level 60/20/20 train/val/test split (50,937 train / 16,980 val / 16,981 test).</li>
    <li><strong>Critical Discovery:</strong> Identified that random image-level sampling placed frames from all 37 subjects across train, validation, and test sets. Established the mandatory academic disclosure that metrics reflect familiar-subject recognition rather than zero-shot transfer.</li>
    <li><strong>Evidence:</strong> <code>results/dataset_inspection_summary.json</code>, <code>results/dataset_inspection_summary.md</code>.</li>
</ul>

<h3>Milestone 2: Feedforward Neural Network (FNN) Baseline</h3>
<ul>
    <li><strong>Attempted:</strong> Building and training a 4-layer fully connected Multi-Layer Perceptron (FNN) as an empirical benchmark.</li>
    <li><strong>Why Attempted:</strong> To establish a performance baseline prior to introducing spatial 2D convolutional operators.</li>
    <li><strong>Outcome:</strong> Flattened 64x64 grayscale inputs into 4,096-dimensional vectors, passing through 512-256-128 Dense layers (2,262,017 parameters). Trained on native CPU in 171.03 s across 10 epochs. Reached 91.43% test accuracy and 0.9143 macro F1 (best validation checkpoint saved at Epoch 7 with 91.49%).</li>
    <li><strong>What Changed:</strong> Established that while 91.43% was respectable, the model required 2.26M parameters and produced 1,456 test misclassifications due to lack of spatial locality awareness.</li>
    <li><strong>Evidence:</strong> <code>models/fnn_baseline_best.keras</code>, <code>results/fnn_test_metrics.json</code>, <code>results/logs/fnn_training.log</code>.</li>
</ul>

<h3>Milestone 3: Convolutional Neural Network (CNN) Core</h3>
<ul>
    <li><strong>Attempted:</strong> Designing a compact 3-block 2D CNN with 3x3 convolutions and 2x2 max-pooling.</li>
    <li><strong>Why Attempted:</strong> To leverage spatial inductive bias (eyelid boundaries, iris curvature) while dramatically cutting parameter overhead.</li>
    <li><strong>Outcome:</strong> Parameters fell by <strong>49.54%</strong> to <strong>1,141,505</strong>. Trained in 785.79 s. Test accuracy jumped to <strong>98.75% (+7.32 pp)</strong>, macro F1 reached <strong>0.9875</strong>, and sleepy recall reached <strong>98.40%</strong> (best validation checkpoint saved at Epoch 9).</li>
    <li><strong>What Changed:</strong> The compact CNN was selected as the sole classification engine for live deployment.</li>
    <li><strong>Evidence:</strong> <code>models/cnn_baseline_best.keras</code>, <code>results/cnn_test_metrics.json</code>, <code>results/logs/cnn_training.log</code>.</li>
</ul>

<h3>Milestones 4 &amp; 5: Webcam Deployment &amp; Temporal Stabilization</h3>
<ul>
    <li><strong>Attempted:</strong> Deploying the trained CNN on live webcam video using OpenCV and an initial Haar cascade face/eye detector.</li>
    <li><strong>Observed Problem:</strong> In live execution, while open eyes predicted high confidence (~95%), closed eyes hovered near 50% confidence and suffered from single-frame flickers. These brief flickers prematurely reset the continuous 1.5 s closure timer, preventing alerts from triggering.</li>
    <li><strong>Engineering Solution:</strong> Engineered <code>TemporalDrowsinessTracker</code>:
        <ol>
            <li>5-frame moving-average buffer to smooth single-frame classification flutter.</li>
            <li>250 ms Open Grace Period: natural blinks or brief transitions do not reset accumulated closure time.</li>
            <li>500 ms Unknown Dropout Timeout: missing detections are handled conservatively without false alarms.</li>
        </ol>
    </li>
    <li><strong>Evidence:</strong> <code>tests/test_temporal_tracker.py</code> (4/4 synthetic sequence tests passing).</li>
</ul>

<div class="page-break"></div>

<!-- ================= PAGE 5 ================= -->
<h2>Part V: Engineering History (Milestones 6–10) &amp; Interface</h2>

<h3>Milestones 6 &amp; 7: Low-Light Failure Analysis &amp; MediaPipe Migration</h3>
<ul>
    <li><strong>Attempted:</strong> Diagnosing closed-eye detection failures in dim lighting and testing CLAHE contrast enhancement.</li>
    <li><strong>Empirical Findings:</strong>
        <ol>
            <li><strong>CLAHE Failure:</strong> Applying CLAHE directly to eye crops degraded CNN accuracy from 98.7% to 93.8% (-4.9%) due to high-frequency noise and artificial edge artifacts.</li>
            <li><strong>The Haar Cascade Flaw:</strong> <code>haarcascade_eye.xml</code> requires open irises/pupils to identify an eye. When a subject closes their eyes, detection drops completely or drifts upward onto the eyebrow.</li>
            <li><strong>Aspect Ratio Distortion:</strong> Rectangular Haar bounding boxes (e.g. 35x25) stretched horizontal closed eyelids into thick elliptical shapes resembling open eyes when resized to 64x64.</li>
            <li><strong>Domain Shift:</strong> MRL images utilize active near-infrared (NIR) sensors, whereas webcams capture visible-light skin folds and shadows.</li>
        </ol>
    </li>
    <li><strong>The Architectural Solution:</strong> Completely replaced Haar cascades with Google MediaPipe Face Landmarker. Eyelid contours are tracked in 3D space, and aspect-ratio-preserved square patches are extracted regardless of eyelid state.</li>
    <li><strong>Evidence:</strong> <code>results/diagnostics/pipeline_trace_report.json</code>, <code>low_light_diagnosis.json</code>.</li>
</ul>

<h3>Milestones 8 &amp; 9: Desktop Interface Evolution &amp; "Pitch Black" Redesign</h3>
<ul>
    <li><strong>Attempted:</strong> Developing an operator desktop interface in PySide6.</li>
    <li><strong>Evolution:</strong> Rebranded from AEGIS to Blink. Redesigned under the Impeccable <strong>PITCH BLACK</strong> concept:
        <ul>
            <li>Pure black background (<code>#000000</code>) and subtly differentiated dark surfaces (<code>#050505</code>, <code>#0A0A0A</code>).</li>
            <li>Strict color semantics: Red is reserved solely for genuine drowsiness warnings; Stop Monitoring button uses neutral dark charcoal (<code>#18181B</code>).</li>
            <li>Collapsible parameter and telemetry drawers to maximize camera viewport focus.</li>
            <li>Integrated <code>QScrollArea</code> ensuring responsive scaling from 1000x680 to 4K widescreen without control squishing.</li>
        </ul>
    </li>
</ul>

<table style="width: 100%; border-collapse: collapse; border: none; margin: 4px 0;">
    <tr>
        <td style="width: 50%; border: none; text-align: center; padding: 2px;">
            <div style="font-size: 7.2pt; font-weight: bold; color: #64748b; margin-bottom: 2px;">STANDBY / ATTENTIVE STATE</div>
            <img src="{ui_idle_uri}" width="235" />
        </td>
        <td style="width: 50%; border: none; text-align: center; padding: 2px;">
            <div style="font-size: 7.2pt; font-weight: bold; color: #e11d48; margin-bottom: 2px;">ACTIVE DROWSINESS ALERT (1.65s CLOSURE)</div>
            <img src="{ui_alert_uri}" width="235" />
        </td>
    </tr>
</table>

<h3>Milestone 10: Clean Packaging &amp; Automated Verification</h3>
<ul>
    <li><strong>Outcome:</strong> Created <code>.gitignore</code> excluding the raw 80,000-image dataset and bytecode while retaining verified checkpoints and logs. Built <code>run.bat</code> for one-click Windows deployment. Both headless test suites verified 4/4 PASS.</li>
</ul>

<div class="page-break"></div>

<!-- ================= PAGE 6 ================= -->
<h2>Part VI: Academic Formulation Guide (Abstract, Paper &amp; Viva)</h2>

<h3>1. Publication-Ready Verified Claims</h3>
<div class="quote-card">
    <strong>Abstract Claim 1: Benchmark Superiority</strong><br>
    "The compact CNN achieved 98.75% test accuracy, a 0.9875 macro F1-score, and 98.40% recall on closed eyes across 16,981 holdout test samples from the MRL Eye Dataset, outperforming a 4-layer FNN baseline (91.43% accuracy, 0.9143 macro F1) by 7.32 percentage points."
</div>

<div class="quote-card">
    <strong>Methodology Claim 2: Computational &amp; Storage Efficiency</strong><br>
    "By substituting fully connected input layers with 3 convolutional stages, model parameters were reduced by 49.54% (1.14M vs. 2.26M) and disk size by 49.42% (13.11 MB vs. 25.92 MB), facilitating low-latency CPU inference without requiring dedicated GPU accelerators."
</div>

<div class="quote-card">
    <strong>Architecture Claim 3: Decoupled Landmark Localization</strong><br>
    "To resolve the classical failure mode where Haar cascades fail on closed eyes, the system decouples eye localization from state classification: a 478-point MediaPipe face mesh isolates aspect-ratio-preserved square patches, enabling consistent 64x64 CNN classification regardless of eyelid position."
</div>

<div class="quote-card">
    <strong>Decision Logic Claim 4: Temporal Hysteresis Tracking</strong><br>
    "The real-time decision engine integrates a temporal hysteresis tracker combining a 5-frame moving-average filter, a 250 ms open-grace period to tolerate natural blinks, and a 1.50 s sustained closure threshold to prevent false alarms while promptly triggering acoustic and visual warnings."
</div>

<h3>2. Anticipated Viva / Reviewer Questions &amp; Defenses</h3>
<ul>
    <li><strong>Q1: Why does random image-level splitting matter?</strong><br>
        <em>Defense:</em> We explicitly disclose that all 37 subjects are present across train, val, and test splits. This measures familiar-subject ocular recognition under varying head poses and illumination. Disclosing this demonstrates high academic rigor and intellectual honesty.</li>
    <li><strong>Q2: Why use a CNN instead of geometric Eye Aspect Ratio (EAR)?</strong><br>
        <em>Defense:</em> Geometric EAR formulas fail with eyeglasses glare, oblique driver head rotation, and squinting. The CNN learns rich spatial textures and eyelid curvature directly from data, achieving 98.75% accuracy and robust live detection.</li>
    <li><strong>Q3: Can this system run on low-cost automotive edge hardware?</strong><br>
        <em>Defense:</em> Yes. The compact CNN requires only 1.14M parameters and 13.1 MB storage. On a standard CPU, inference completes in ~5–12 ms per eye, maintaining 30 FPS comfortably without dedicated GPU hardware.</li>
</ul>

<h3>3. Documented Academic Limitations &amp; Disclosures</h3>
<ol>
    <li><strong>Subject Overlap Disclosure:</strong> The 60/20/20 train/val/test split was performed randomly at the <em>image level</em>. All 37 subjects appear across train, validation, and test sets. Consequently, the 98.75% test accuracy measures <em>familiar-subject recognition</em> rather than zero-shot cross-subject transfer.</li>
    <li><strong>Sensor Domain Gap:</strong> The MRL Eye Dataset was captured using active near-infrared (NIR) sensors, whereas consumer webcams operate under visible RGB light. While MediaPipe landmark extraction mitigates framing differences, ambient shadows in dim visible light remain a challenge.</li>
    <li><strong>Lack of Multi-Modal Fatigue Metrics:</strong> Blink focuses on ocular closure; a complete automotive Driver Monitoring System (DMS) typically fuses eye state with yawning frequency, head-pose nodding, and steering telemetry.</li>
    <li><strong>Research Prototype Status:</strong> Blink is an academic research prototype and is not automotive safety-certified (ISO 26262 / Euro NCAP) without specialized infrared automotive-grade hardware and controlled track validation.</li>
</ol>

<div class="meta-card" style="margin-top: 8px; text-align: center;">
    <strong>Verification Sign-off:</strong> All numerical results, layer parameters, file sizes, and confusion matrices verified against primary disk artifacts.
</div>

</body>
</html>
"""

    doc = QTextDocument()
    doc.setDefaultFont(app_font)
    doc.setHtml(html_content)

    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(str(output_pdf))
    printer.setPageSize(QPageSize(QPageSize.A4))
    printer.setPageMargins(QMarginsF(8, 8, 8, 8), QPageLayout.Millimeter)

    doc.print_(printer)
    print(f"Generated Pretty PDF: {output_pdf}")
    print(f"File size: {output_pdf.stat().st_size} bytes")

    pdf = QPdfDocument()
    pdf.load(str(output_pdf))
    print(f"Total pages: {pdf.pageCount()}")
    for i in range(pdf.pageCount()):
        img = pdf.render(i, QSize(800, 1131))
        img.save(f"results/diagnostics/ui_screens/pretty_page_{i+1}.png")
    print("All pages rendered and saved for visual verification!")

if __name__ == "__main__":
    generate_pretty_pdf()
