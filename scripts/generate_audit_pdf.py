import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QTextDocument, QPageLayout, QPageSize
from PySide6.QtCore import QMarginsF
from PySide6.QtPrintSupport import QPrinter

def generate_pdf():
    app = QApplication.instance() or QApplication([])

    output_pdf = Path("Blink_Rigorous_Factual_Audit_Report.pdf").resolve()

    html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.45;
        color: #1e293b;
        margin: 0;
        padding: 0;
    }
    h1 {
        font-size: 18pt;
        color: #0f172a;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 8px;
    }
    h2 {
        font-size: 13.5pt;
        color: #1e3a8a;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 4px;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    h3 {
        font-size: 11.5pt;
        color: #0f172a;
        margin-top: 14px;
        margin-bottom: 6px;
    }
    .meta-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 14px;
        font-size: 9pt;
        color: #475569;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0 14px 0;
        font-size: 9pt;
    }
    th, td {
        border: 1px solid #cbd5e1;
        padding: 6px 8px;
        text-align: left;
    }
    th {
        background-color: #f1f5f9;
        font-weight: 700;
        color: #0f172a;
    }
    tr:nth-child(even) {
        background-color: #f8fafc;
    }
    .badge-verified {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 8pt;
    }
    .badge-warning {
        display: inline-block;
        background-color: #fef3c7;
        color: #b45309;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 8pt;
    }
    .alert-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 8px 12px;
        margin: 10px 0;
        font-size: 9.5pt;
        color: #991b1b;
    }
    pre, code {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 8.5pt;
    }
    .code-box {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 8px 0;
        line-height: 1.35;
    }
    ul, ol {
        margin-top: 4px;
        margin-bottom: 10px;
        padding-left: 20px;
    }
    li {
        margin-bottom: 4px;
    }
    .quote-card {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 8px 12px;
        margin: 8px 0;
        font-style: italic;
        color: #1e40af;
        font-size: 9.5pt;
    }
    .page-break {
        page-break-before: always;
    }
</style>
</head>
<body>

<h1>Blink Project: Rigorous Factual Audit &amp; Technical Narrative</h1>

<div class="meta-box">
    <strong>Document Purpose:</strong> Academic Verification Baseline (Abstract, Methodology, Results, and Viva Preparation)<br>
    <strong>Project Title:</strong> Blink | Real-Time Driver Drowsiness Detection System<br>
    <strong>Audit Date:</strong> October 20, 2026 &bull; <strong>Host OS:</strong> Windows 10/11 (AMD64) &bull; <strong>Hardware Target:</strong> CPU Execution<br>
    <strong>Audited Artifacts:</strong> <code>src/</code>, <code>models/</code>, <code>scripts/</code>, <code>results/</code>, <code>results/logs/</code>, <code>tests/</code>
</div>

<h2>Part I: Factual Audit &amp; System Verification</h2>

<h3>1. Project Identity &amp; System Architecture</h3>
<table>
    <tr>
        <th style="width: 20%;">Component</th>
        <th style="width: 35%;">Claimed Specification</th>
        <th style="width: 35%;">Verified Artifact / Code Source</th>
        <th style="width: 10%;">Status</th>
    </tr>
    <tr>
        <td><strong>Project Identity</strong></td>
        <td>Blink (formerly AEGIS)</td>
        <td>Window title: <code>"Blink | Driver Monitoring"</code> in <code>src/desktop_app.py:276</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Primary Objective</strong></td>
        <td>Real-time ocular classification &amp; drowsiness detection</td>
        <td>64x64 CNN inference + temporal accumulation in <code>src/webcam_detector.py</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Vision Frontend</strong></td>
        <td>MediaPipe Face Landmarker (478 landmarks)</td>
        <td>Model: <code>models/face_landmarker.task</code> (3.58 MB) via <code>src/eye_extractor.py</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Classification Core</strong></td>
        <td>TensorFlow/Keras Compact CNN</td>
        <td>Model: <code>models/cnn_baseline_best.keras</code> (13.11 MB, 1.14M params)</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Desktop UI</strong></td>
        <td>PySide6 Pitch Black Desktop Interface</td>
        <td>Asynchronous <code>QThread</code> worker, thread-safe Qt Signals in <code>src/desktop_app.py</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Alert Mechanism</strong></td>
        <td>2.5 kHz audio buzzer + high-urgency UI banner</td>
        <td><code>winsound.Beep(2500, 250)</code> with cooldown in <code>src/webcam_detector.py:131</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
</table>

<h3>2. Dataset Properties &amp; Split Integrity (MRL Eye Dataset)</h3>
<table>
    <tr>
        <th>Attribute</th>
        <th>Verified Metric / Value</th>
        <th>Primary Evidence File</th>
        <th>Status</th>
    </tr>
    <tr>
        <td><strong>Dataset Source</strong></td>
        <td>MRL Eye Dataset (Large-scale ocular repository)</td>
        <td><code>results/dataset_inspection_summary.json:2</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Total Images</strong></td>
        <td><strong>84,898 images</strong> (0 corrupted / unreadable)</td>
        <td><code>results/dataset_inspection_summary.json:3,31</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Native Image Format</strong></td>
        <td>Grayscale (Mode 'L'), variable resolutions (e.g. 84x84, 82x82)</td>
        <td><code>results/dataset_inspection_summary.json:33-47</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Preprocessed Format</strong></td>
        <td>64x64 single-channel grayscale, normalized [0, 1]</td>
        <td><code>src/data_loader.py:7</code> &amp; <code>Rescaling(1./255)</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Class Distribution</strong></td>
        <td>Awake: 42,952 (50.59%) | Sleepy: 41,946 (49.41%)</td>
        <td><code>results/dataset_inspection_summary.json:23-30</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Data Partitioning</strong></td>
        <td>Train: 50,937 (60%) | Val: 16,980 (20%) | Test: 16,981 (20%)</td>
        <td><code>results/dataset_inspection_summary.json:18-22</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Class Mapping</strong></td>
        <td>Index 0 = <code>sleepy</code> (closed) | Index 1 = <code>awake</code> (open)</td>
        <td><code>src/data_loader.py:6</code> (<code>CLASS_NAMES</code>)</td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
    <tr>
        <td><strong>Subject Overlap</strong></td>
        <td>All 37 subjects present in Train, Val, and Test splits</td>
        <td><code>results/dataset_inspection_summary.json:48-55</code></td>
        <td><span class="badge-verified">VERIFIED</span></td>
    </tr>
</table>

<h3>3. Model Architecture &amp; Parameter Verification</h3>
<div class="code-box">
<pre>
FNN Baseline ("FNN_Baseline"):
  Layer 1: Input (None, 64, 64, 1) -> Rescaling(1./255)
  Layer 2: Flatten -> 4096 units
  Layer 3: Dense(512, relu) -> Dropout(0.3)   [Params: 2,097,664]
  Layer 4: Dense(256, relu) -> Dropout(0.3)   [Params:   131,328]
  Layer 5: Dense(128, relu) -> Dropout(0.3)   [Params:    32,896]
  Layer 6: Dense(1, sigmoid)                   [Params:       129]
  Total Parameters: 2,262,017 (100% trainable) | Size on disk: 25.92 MB

CNN Core ("CNN_Baseline"):
  Block 1: Conv2D(32, 3x3, same, relu) + MaxPool2D(2x2) -> (None, 32, 32, 32) [Params:       320]
  Block 2: Conv2D(64, 3x3, same, relu) + MaxPool2D(2x2) -> (None, 16, 16, 64) [Params:    18,496]
  Block 3: Conv2D(128, 3x3, same, relu) + MaxPool2D(2x2)-> (None,  8,  8, 128)[Params:    73,856]
  Classifier: Flatten (8192) -> Dense(128, relu) -> Dropout(0.3)               [Params: 1,048,704]
  Output:     Dense(1, sigmoid)                                                [Params:       129]
  Total Parameters: 1,141,505 (100% trainable) | Size on disk: 13.11 MB
</pre>
</div>

<div class="alert-box">
    <strong>Critical Documentation Discrepancy Corrected:</strong><br>
    The README originally stated <em>"4-Block Conv2D + BatchNorm + Dense"</em>. Direct inspection of <code>src/cnn_model.py</code> and <code>models/cnn_baseline_best.keras</code> proves the model actually has <strong>3 Conv2D blocks, 0 BatchNormalization layers, and 1 Dense layer</strong>.
</div>

<div class="page-break"></div>

<h2>Part II: Verified Comparative Results ($N = 16,981$ Test Samples)</h2>

<table>
    <tr>
        <th>Performance Metric</th>
        <th>Baseline FNN</th>
        <th>Compact CNN (Blink Core)</th>
        <th>Observed Absolute Delta</th>
    </tr>
    <tr>
        <td><strong>Model Architecture</strong></td>
        <td>4-Layer Dense MLP</td>
        <td>3-Block Conv2D + Dense</td>
        <td>Spatial Inductive Bias</td>
    </tr>
    <tr>
        <td><strong>Trainable Parameters</strong></td>
        <td>2,262,017</td>
        <td><strong>1,141,505</strong></td>
        <td><strong>-49.54% (-1,120,512 params)</strong></td>
    </tr>
    <tr>
        <td><strong>Model File Size</strong></td>
        <td>25.92 MB</td>
        <td><strong>13.11 MB</strong></td>
        <td><strong>-49.42% (-12.81 MB)</strong></td>
    </tr>
    <tr>
        <td><strong>Training Time (CPU)</strong></td>
        <td>171.03 s (~2.85 min)</td>
        <td>785.79 s (~13.10 min)</td>
        <td>+359.4% (4.6x training duration)</td>
    </tr>
    <tr>
        <td><strong>Best Val Checkpoint</strong></td>
        <td>Epoch 7 (val_acc: 91.49%)</td>
        <td>Epoch 9 (val_acc: 98.75%)</td>
        <td>Saved via <code>val_accuracy</code> monitor</td>
    </tr>
    <tr>
        <td><strong>Test Accuracy</strong></td>
        <td>91.43% (15,525 / 16,981)</td>
        <td><strong>98.75% (16,768 / 16,981)</strong></td>
        <td><strong>+7.32 percentage points</strong></td>
    </tr>
    <tr>
        <td><strong>Macro F1-Score</strong></td>
        <td>0.9143</td>
        <td><strong>0.9875</strong></td>
        <td><strong>+0.0732</strong></td>
    </tr>
    <tr>
        <td><strong>Sleepy Recall (Closed)</strong></td>
        <td>92.60% (7,769 / 8,390)</td>
        <td><strong>98.40% (8,256 / 8,390)</strong></td>
        <td><strong>+5.80 percentage points</strong></td>
    </tr>
    <tr>
        <td><strong>Awake Recall (Open)</strong></td>
        <td>90.28% (7,756 / 8,591)</td>
        <td><strong>99.08% (8,512 / 8,591)</strong></td>
        <td><strong>+8.80 percentage points</strong></td>
    </tr>
    <tr>
        <td><strong>Sleepy Precision</strong></td>
        <td>90.30%</td>
        <td><strong>99.05%</strong></td>
        <td>+8.75 percentage points</td>
    </tr>
    <tr>
        <td><strong>Awake Precision</strong></td>
        <td>92.59%</td>
        <td><strong>98.45%</strong></td>
        <td>+5.86 percentage points</td>
    </tr>
</table>

<h3>Ground-Truth Test Confusion Matrices ($N = 16,981$)</h3>
<div class="code-box">
<pre>
FNN Test Matrix (results/fnn_test_metrics.json):
                    Predicted Sleepy (0)   Predicted Awake (1)
  Actual Sleepy (0):        7,769                  621          (Recall: 92.60%)
  Actual Awake  (1):          835                7,756          (Recall: 90.28%)

CNN Test Matrix (results/cnn_test_metrics.json):
                    Predicted Sleepy (0)   Predicted Awake (1)
  Actual Sleepy (0):        8,256                  134          (Recall: 98.40%)
  Actual Awake  (1):           79                8,512          (Recall: 99.08%)
</pre>
</div>

<h2>Part III: Verified Real-Time Vision Pipeline</h2>

<ul>
    <li><strong>MediaPipe Decoupled Extraction:</strong> Replaced classic Haar eye cascades with a 478-landmark Face Landmarker. Extracts square eye bounding boxes from inner and outer eye corners (landmarks 33/133 and 362/263) with a 1.35x margin. Evaluates eyelids robustly whether eyes are open or closed.</li>
    <li><strong>Aspect-Ratio Preservation:</strong> Guarantees a strict 1:1 aspect ratio prior to 64x64 resizing, eliminating horizontal eyelid stretching.</li>
    <li><strong>Temporal Hysteresis Engine (<code>TemporalDrowsinessTracker</code>):</strong>
        <ul>
            <li><strong>5-Frame Rolling Average:</strong> Smooths raw sigmoid predictions across consecutive frames to suppress single-frame classification noise.</li>
            <li><strong>250 ms Open Grace Period:</strong> Natural blinks or momentary flickers back to open state do not reset accumulated closure time.</li>
            <li><strong>500 ms Unknown Timeout:</strong> Facial tracking dropouts are handled conservatively without triggering false alarms.</li>
            <li><strong>1.50 s Sustained Closure Threshold:</strong> Triggers audible alarm and visible warning banner once continuous closure duration crosses 1.50 s.</li>
        </ul>
    </li>
    <li><strong>Verified Test Suites:</strong> 
        <ul>
            <li><code>tests/test_desktop_app_headless.py</code>: <strong>4/4 PASS</strong></li>
            <li><code>tests/test_temporal_tracker.py</code>: <strong>4/4 PASS</strong></li>
        </ul>
    </li>
</ul>

<div class="page-break"></div>

<h2>Part IV: Reconstructed Project Development Narrative</h2>

<h3>Milestone 1: Problem Definition &amp; Dataset Selection</h3>
<ul>
    <li><strong>Attempted:</strong> Developing an ocular classification engine on the MRL Eye Dataset (84,898 images across 37 human subjects).</li>
    <li><strong>Outcome:</strong> Verified zero unreadable files and balanced classes (50.59% awake, 49.41% sleepy). Established an image-level 60/20/20 split.</li>
    <li><strong>Key Discovery:</strong> Identified that random image-level partitioning distributed all 37 subjects across train, validation, and test sets. Established the mandatory academic disclosure that metrics evaluate familiar-subject recognition rather than zero-shot transfer.</li>
</ul>

<h3>Milestone 2: FNN Baseline Development</h3>
<ul>
    <li><strong>Attempted:</strong> Training a flat Multi-Layer Perceptron (FNN) to establish an empirical performance floor.</li>
    <li><strong>Outcome:</strong> Reached 91.43% test accuracy and 0.9143 macro F1 in 171.03 s CPU training time, but required 2.26M parameters and produced 1,456 misclassifications on the test set.</li>
</ul>

<h3>Milestone 3: CNN Core Development</h3>
<ul>
    <li><strong>Attempted:</strong> Designing a compact 3-block 2D CNN to capture spatial ocular geometry.</li>
    <li><strong>Outcome:</strong> Parameter count fell by 49.54% (1.14M vs. 2.26M) while test accuracy rose to <strong>98.75% (+7.32 pp)</strong> and sleepy recall reached <strong>98.40%</strong>. ModelCheckpoint saved the optimal weights at Epoch 9.</li>
</ul>

<h3>Milestone 4 &amp; 5: Webcam Deployment &amp; Temporal Stabilization</h3>
<ul>
    <li><strong>Attempted:</strong> Integrating the CNN with OpenCV webcam capture.</li>
    <li><strong>Observed Problem:</strong> In live testing, closed eyes hovered around 50% confidence with brief flickers that reset the sustained closure timer, preventing alerts from firing.</li>
    <li><strong>Solution:</strong> Developed <code>TemporalDrowsinessTracker</code> introducing a 5-frame moving average and a 250 ms open-grace period, verified across synthetic sequence tests.</li>
</ul>

<h3>Milestone 6 &amp; 7: Low-Light Failure Analysis &amp; MediaPipe Migration</h3>
<ul>
    <li><strong>Attempted:</strong> Diagnosing closed-eye detection failures in dim lighting and testing CLAHE contrast enhancement.</li>
    <li><strong>Empirical Discovery:</strong> CLAHE degraded CNN accuracy by 4.9% (98.7% -> 93.8%) due to contrast artifacts. The true root cause was that <code>haarcascade_eye.xml</code> requires open pupils/irises to detect eyes, failing completely on closed eyelids or drifting to the eyebrow. Furthermore, rectangular Haar crops distorted eyelid aspect ratios.</li>
    <li><strong>Solution:</strong> Replaced Haar cascades with MediaPipe Face Landmarker, tracking 3D facial landmarks and extracting aspect-ratio-preserved square eye patches regardless of eyelid state.</li>
</ul>

<h3>Milestone 8 &amp; 9: Desktop Application Cockpit &amp; "Pitch Black" Polish</h3>
<ul>
    <li><strong>Attempted:</strong> Building an operator desktop cockpit using PySide6.</li>
    <li><strong>Outcome:</strong> Rebranded from AEGIS to Blink. Redesigned under the Impeccable "Pitch Black" concept: near-pure black background (<code>#000000</code>), white typography, collapsible parameter/telemetry panels, responsive <code>QScrollArea</code>, and strict reservation of red only for genuine drowsiness emergencies.</li>
</ul>

<h2>Part V: Academic Formulation: Verified Claims vs. Limitations</h2>

<h3>Safe Claims for Academic Papers &amp; Presentations</h3>
<div class="quote-card">
    &bull; "The compact CNN achieved 98.75% test accuracy, 0.9875 macro F1, and 98.40% sleepy recall on 16,981 holdout test samples, surpassing the 4-layer FNN baseline by 7.32 percentage points."
</div>
<div class="quote-card">
    &bull; "By replacing dense input layers with 3 convolutional stages, model parameter count was reduced by 49.5% (1.14M vs. 2.26M params) and disk size by 49.4% (13.11 MB vs. 25.92 MB), enabling responsive native CPU execution."
</div>
<div class="quote-card">
    &bull; "Decoupling eye localization from eye-state classification via 478-point MediaPipe face mesh geometry solved the fundamental failure mode of classic Haar cascades on closed eyelids."
</div>

<h3>Documented Academic Limitations</h3>
<ol>
    <li><strong>Subject Overlap:</strong> The 60/20/20 image-level split shares subjects across splits; test accuracy reflects within-subject generalization rather than zero-shot cross-subject transfer.</li>
    <li><strong>Sensor Domain Gap:</strong> MRL images were collected under active near-infrared (NIR) sensors, whereas consumer webcams operate in visible RGB light.</li>
    <li><strong>Non-Certified Status:</strong> Blink is an academic research prototype and is not automotive safety-certified (ISO 26262 / Euro NCAP) without automotive-grade infrared hardware.</li>
</ol>

</body>
</html>
"""

    doc = QTextDocument()
    doc.setHtml(html_content)

    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(str(output_pdf))
    printer.setPageSize(QPageSize(QPageSize.A4))
    printer.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout.Millimeter)

    doc.print_(printer)
    print(f"Generated PDF: {output_pdf}")
    print(f"File size: {output_pdf.stat().st_size} bytes")

if __name__ == "__main__":
    generate_pdf()
