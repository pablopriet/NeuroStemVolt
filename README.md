<p align="center">
  <img src="https://github.com/user-attachments/assets/06242e82-c960-4d6b-a0eb-0c749dd6590f" alt="NeuroStemVolt Logo" width="500"/>
</p>


# **NeuroStemVolt**

**NeuroStemVolt** is a user-friendly analysis tool with a graphical interface for processing **fast-scan cyclic voltammetry (FSCV)** data from **iPSC-derived neuronal spheroids**.  
It enables in-depth analysis of:

- Neuronal excitability  
- Transporter kinetics  
- Drug response dynamics over time

Whether you're characterizing iPSC-derived neuronal systems or investigating neurotransmission under pharmacological conditions, **NeuroStemVolt** offers streamlined and flexible analysis workflows that adapt to a wide range of experimental setups in modern neurophysiological research.

---

## **Key Features**

### **Data Pre-processing**
- Interactive color plots and current-vs-time traces  
- Multiple filtering options:
  - Baseline correction
  - Background subtraction
  - Rolling mean
  - Butterworth filtering
  - Savitzky-Golay
  - Normalization
- Visualization of raw vs. filtered data
- Non-specific waveform support

### **Stimulation Analysis**
- Stimulation time markers overlaid on current-vs-time traces
- Configurable stimulation parameters (onset, duration, amplitude) loaded from experiment settings
- Stimulation artifact removal with adjustable parameters
- Per-replicate stimulation event annotation

### **Spontaneous Activity Detection**
- Detection and quantification of spontaneous neurotransmitter release events  
- Peak detection algorithms with configurable thresholds
- **Manual peak editing** — add, remove, or adjust detected peaks via interactive editor
- Metadata flagging to track which files have had peaks manually modified

### **Neuronal Excitability Analysis**
- Analysis of stimulus-evoked release amplitudes  
- Assessment of neuronal responsiveness
- Exponential decay fitting **per replicate**, with pooled statistics across replicates

### **Transporter Kinetics Evaluation**
- Quantification of reuptake kinetics  
- Inference of neurotransmitter transporter activity
- Global and per-replicate exponential fitting with pooled summary statistics

### **Drug Response Analysis**
- Comparison of release amplitude and clearance rates  
- Across time series of measurements

### **Export Tools**
- Export processed and annotated results to `.csv` format  
- **Session export** covers all replicates and all detected/edited peaks in one operation
- Experiment log (notes, settings, metadata) included in CSV export
- Ideal for downstream statistical analysis and sharing

### **Graphical User Interface (GUI)**
- No coding required  
- Intuitive workflows for:
  - Analysis setup  
  - Filtering  
  - Visualization  
  - Export  
- Built-in support for batch processing
- Progress and confirmation dialogs for long-running or destructive operations
- Context-sensitive help dialogs throughout the interface
- OS-adaptive text and UI styling

---

## **What's New in v1.1.1**
- **Multi-peak detection overhaul** — all FindAmplitudeMultiple detector parameters are now configurable through from_params() for complete control over detection behaviour
- **Selective exponential-fit plotting** — choose exactly which timepoints generate exponential-fit figures using a new multi-select dialog
- **Cleaner multi-peak results** — removed the redundant Peak Amplitudes plot and hidden Mean IT plots in Multi-Peak mode to reduce duplicate output
- **Improved exponential fitting** — unified the fitting workflow, corrected joint fitting of the A and C parameters, and improved plot axes with adaptive tick spacing
- **Per-timepoint exponential exports** — exponential-fit plots are now exported individually instead of overwriting a single figure
- **Treatment-aware CSV exports** — treatment names are now automatically included in exported CSV filenames
- **Windowed amplitude plot** — added a new windowed amplitude plot to improve visualisation of signal changes over time
- **Improved plotting consistency** — all time-based plots now display and export in seconds using a shared waveform voltage-axis helper
- **SVG export support** — figures can now be exported as scalable vector graphics (SVG) in addition to existing formats
- **Improved CV exports** — fixed the cyclic voltammogram voltage axis, included CV plots in Save Current Plots, and consolidated all exported plots into a single plots folder
- **Manual peak editor restored** — interactive click-to-edit peak selection has been re-enabled
- **Improved help text** — experiment settings documentation has been clarified
- **Repository updates** — refreshed example datasets, updated example outputs, and cleaned up internal documentation and source code comments

---

## **Input Format**

- **Supported Input:** `.txt` files containing color plot data (one per timepoint), formatted as tab- or space-separated values.
- **How to Prepare Data:** Place all files for a replicate in a single folder.  
- **Formatting Guidance:** See the `example_data` folder for sample file structure and formatting tips.

---

## **Applications**

- Functional characterization of iPSC-derived neuronal spheroids
- Analysis of neurotransmitter release and reuptake under pharmacological manipulation
- Drug screening, dose-response profiling, and transporter kinetics studies
- Exploratory research in neurochemical signaling and synaptic physiology

---

## **Output**

**CSV Exports:**
- Filtered and processed current-vs-time traces for each replicate and timepoint
- Detected peak events with timestamps, amplitudes, and edit-status flags
- Reuptake curve parameters: per-replicate fits, pooled statistics, decay constants
- Experiment log and session metadata

**Visualizations:**
- Interactive color plots for each timepoint
- Current-vs-time traces with stimulation markers and event annotations
---

## Running NeuroStemVolt

You can either use the pre-built executables (no Python setup required) or run directly from source.

### Option 1: Pre-built Executables

Pre-built executables are available for Windows and macOS.  
These versions run without requiring any Python installation or environment setup.

Download the latest release here:  
[**NeuroStemVolt v1.1.1 – Release Assets**](https://github.com/pablopriet/NeuroStemVolt/releases/tag/v1.1.1)

### Included in the release:
- `NeuroStemVolt-windows-v1.1.1.exe` — Windows executable
- `NeuroStemVolt-mac-v1.1.1.zip` — macOS application bundle (`.app`)

**Usage:**
1. Download the file for your operating system.
2. On **Windows**:
   - Double-click the `.exe` to run the program.
   - If you see a SmartScreen prompt, click **More info → Run anyway**.
3. On **macOS**:
   - Unzip the downloaded file.
   - Double-click `NeuroStemVolt.app` to launch.
   - If you see a security warning, go to **System Settings → Privacy & Security** and click **Open Anyway**.
   - If macOS says the app is "damaged", open Terminal and run:  
     ```bash
     xattr -cr /path/to/NeuroStemVolt.app
     ```

### Option 2: Run from Source

#### Method A – Using pip
**Requirements**
- Python 3.11+
- numpy, pandas, matplotlib, PyQt5, scipy

**1) Clone the repository**
```bash
git clone https://github.com/pablopriet/NeuroStemVolt.git
cd NeuroStemVolt
```

**2) Install dependencies**
```bash
pip install -r requirements.txt
```

**3) Launch the app**
```bash
python main.py
```

---

#### Method B – Using `environment.yml` (Recommended for Conda Users)
**Requirements**
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)

**1) Clone the repository**
```bash
git clone https://github.com/pablopriet/NeuroStemVolt.git
cd NeuroStemVolt
```

**2) Create and activate the environment**
```bash
conda env create -f environment.yml
conda activate neurostemvolt
```

**3) Launch the app**
```bash
python main.py
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-xyz`).
3. Commit your changes.
4. Push to your fork and submit a pull request.

---

## Dependencies

Minimal environment (see `environment.yml` or `requirements.txt`):

- Python **3.11** (tested with 3.11.6)
- numpy **2.1.3**
- pandas **2.2.3**
- matplotlib **3.10.0**
- pyqt **5.15.9**  *(pip users install `PyQt5==5.15.9`)*
- scipy **1.15.1**


## License

[MIT License](LICENSE)  
© 2026 Hashemi Lab · NeuroStemVolt

<p align="center">
  <img width="402" height="202" alt="Hashemi Lab neuron logo design" src="https://github.com/user-attachments/assets/cdae03b3-0f10-4e09-8647-0d591bba7502" />
</p>

---

## Acknowledgements

Developed by **Pablo Prieto Roca** and Tomas Andriuskevicius @[Hashemi Lab](https://www.hashemilab.com/).  
For questions or support, contact [pablo.prieto-roca23@imperial.ac.uk].
