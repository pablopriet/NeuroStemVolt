<p align="center">
  <img src="https://github.com/user-attachments/assets/06242e82-c960-4d6b-a0eb-0c749dd6590f" alt="NeuroStemVolt Logo" width="500"/>
</p>


# **NeuroStemVolt**

**NeuroStemVolt** is a user-friendly analysis tool with a graphical interface for processing **fast-scan cyclic voltammetry (FSCV)** data from **iPSC-derived neuronal spheroids**.  
It enables in-depth analysis of:

- Spontaneous neurotransmitter release  
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
  - Background subtraction, 
  - Rolling mean, 
  - Butterworth filtering 
  - Savitzky-Golay
  - Normalization.
- Visualization of raw vs. filtered data

### **Spontaneous Activity Detection**
- Detection and quantification of spontaneous neurotransmitter release events  
- Peak detection algorithms

### **Neuronal Excitability Analysis**
- Analysis of stimulus-evoked release amplitudes  
- Assessment of neuronal responsiveness

### **Transporter Kinetics Evaluation**
- Quantification of reuptake kinetics  
- Inference of neurotransmitter transporter activity

### **Drug Response Analysis**
- Comparison of release amplitude and clearance rates  
- Across time series of measurements

### **Export Tools**
- Export processed and annotated results to `.csv` format  
- Ideal for downstream statistical analysis and sharing

### **Graphical User Interface (GUI)**
- No coding required  
- Intuitive workflows for:
  - Analysis setup  
  - Filtering  
  - Visualization  
  - Export  
- Built-in support for batch processing

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
- Detected peak events with timestamps and amplitudes
- Reuptake curve parameters, including decay constants and fit statistics

**Visualizations:**
- Interactive color plots for each timepoint
- Current-vs-time traces with event annotations
- Summary plots for batch analysis and group

---

## Running NeuroStemVolt

You can either use the pre-built executables (no Python setup required) or run directly from source.

### Option 1: Pre-built Executables

Pre-built executables are available for macOS and Unix-based systems.  
These versions run without requiring any Python installation or environment setup.

You can download them here:  
[**NeuroStemVolt v1.0.0 – Release Assets**](https://github.com/pablopriet/NeuroStemVolt/releases/tag/v1.0.0)

### Included in the ZIP:
- `.app` – macOS application bundle
- Unix executable – for Linux or macOS terminal use

**Usage:**
1. Download and unzip the file for your operating system.
2. On macOS:
   - Double-click the `.app` to launch the GUI.
   - If you get a security warning, go to **System Preferences → Security & Privacy** and allow the app.
3. On Unix/Linux:
   - Open a terminal in the extracted folder.
   - Run:  
     ```bash
     ./NeuroStemVolt
     ```

Windows executables will be provided in a future release.

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
- **Option 1 (Recommended)**
```bash
pip install -r requirements.txt
```
- **Option 2** 
```bash
pip install numpy pandas matplotlib PyQt5 scipy
```


**3) Launch the app**
```bash
python main.py
```

This will start the NeuroStemVolt GUI using your local Python environment.

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

This will start the NeuroStemVolt GUI in a fully managed conda environment.

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
© 2025 Hashemi Lab · NeuroStemVolt

---

## Acknowledgements

Developed by **Pablo Prieto Roca** and Tomas Andriuskevicius @[Hashemi Lab](https://www.hashemilab.com/).  
For questions or support, contact [pablo.prieto-roca23@imperial.ac.uk].


