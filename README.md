# 🛡️ Real-Time Biometric Access Control & Attendance Logging Core Suite

A production-grade, lightweight Computer Vision application engineered in Python to facilitate automated, contactless facial recognition and attendance registration. Utilizing optimized localized edge detection matrices and texture extraction classifiers, this terminal-driven suite bypasses heavy UI overheads to maintain a seamless inference loop.

---

## 🚀 Key Features & Architectural Highlights

*   **Optimized Face Detection:** Implements a tuned **Haar Cascade Classifier** (`scaleFactor=1.05`, `minNeighbors=8`) to strictly isolate sharp facial geometries and suppress false-positive edge noises.
*   **Localized Texture Recognition:** Powered by the **LBPH (Local Binary Patterns Histograms)** Face Recognizer algorithm, enabling fast mathematical pixel training and weight synchronization without external cloud dependencies.
*   **Non-Volatile Analytics Engine:** Automatically serializes authorized check-ins into an structured, duplicate-free `attendance.csv` sheet containing precise `Name, Date, Time` stamps.
*   **Secure Data Purge Core:** Includes an administrative data purge mechanism that cleanly wipes selective biometric files from local disks and instantly auto-calibrates the trained weights.

---

## 🛠️ System Requirements & Dependencies

Ensure your environment is running Python 3.x. The suite strictly utilizes standard analytical and computer vision libraries:

```bash
pip install opencv-contrib-python numpy

Note: Ensure opencv-contrib-python is installed rather than the base package to provide access to the extended cv2.face LBPH module.


📂 Project Directory Structure
├── known_faces/                     # Local biometric database (Face Sample Grayscales)
├── haarcascade_frontalface_default.xml  # Core XML cascade feature model
├── main.py                          # Master control executable script (CLI Panel)
├── trainer.yml                      # Trained local network weights (Auto-Generated)
├── labels.txt                       # Synchronized User ID maps (Auto-Generated)
└── attendance.csv                   # Non-volatile analytics ledger (Auto-Generated)

⚙️ How To Run & Operate
Execute the master script from your terminal root directory:
python main.py

Administrative Operations Overview:
  1.📸 Register New Face (Option 1): Ingests an alphanumeric name, triggers a continuous webcam sampling window, isolates 15 structural grayscale matrix crops, and flashes local training routines.
  2.▶ Launch Inference Stream (Option 2): Fires up the real-time matching HUD overlay. If a subject drops below the defined strict confidence threshold ($< 65$), check-ins are logged locally. Unregistered identities trigger an explicit "Unknown Face" warning indicator.
  3.🗑️ Delete Biometric Data (Option 3): Completely purges target profile data structures, updates indices, and cleanly re-compiles the LBPH weights matrix file.
📊 Telemetry Output Sample
The generated attendance.csv provides a highly accurate logging sequence compatible with analytical spreadsheet pipelines:

Name        Date            Time
User_Sample 2026-07-05    17:45:12



