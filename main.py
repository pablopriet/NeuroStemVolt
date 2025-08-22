from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QSplashScreen, QApplication, QWizard
import sys
import os

from ui.wizard_1_intro.intro_page import IntroPage
from ui.wizard_2_colorplot.colorplot_page import ColorPlotPage
from ui.wizard_3_results.results_page import ResultsPage

def main():
    """
    Launches the NeuroStemVolt Qt application.

    This function initializes the Qt environment, displays a splash screen,
    and loads the multi-page wizard interface for processing FSCV replicates.
    """
    # Enable high DPI to show loading image in high quality
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Load and scale the splash image
    base = os.path.dirname(__file__)
    logo_path = os.path.join(base, 'ui/graphics/LogoNeuroStemVoltV1.0.0.png')
    
    # Check if file exists
    if not os.path.exists(logo_path):
        print(f"Warning: Logo file not found at {logo_path}")
    
    pixmap = QPixmap(logo_path)
    
    # Get original image dimensions
    orig_w, orig_h = pixmap.width(), pixmap.height()
    print(f"Original image size: {orig_w}x{orig_h}")
    
    # Scale the image given the device pixels
    screen = app.primaryScreen()
    sw, sh = screen.size().width(), screen.size().height()
    dpr = screen.devicePixelRatio()

    # Scale image 
    scale = 1 / (12 ** 0.5)  
    tgt_w = int(sw * scale)
    tgt_h = int(sh * scale)
    
    print(f"Target size: {tgt_w}x{tgt_h}, DPR: {dpr}")
    
    # Only scale if the target size is smaller than original, to prevent upscaling
    if tgt_w < orig_w or tgt_h < orig_h:
        scaled = pixmap.scaled(
            int(tgt_w * dpr), int(tgt_h * dpr), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        scaled.setDevicePixelRatio(dpr)  # Set the DPR for proper rendering
    else:
        scaled = pixmap
    
    # Create splash screen
    splash = QSplashScreen(scaled, Qt.WindowStaysOnTopHint)
    
    # Center the splash screen
    geo = splash.frameGeometry()
    center = screen.availableGeometry().center()
    geo.moveCenter(center)
    splash.move(geo.topLeft())
    
    # Show splash and force immediate update
    splash.show()
    app.processEvents()  # Force the splash to render immediately
    
    # Create wizard (but don't show yet)
    wizard = QWizard()
    wizard.setWizardStyle(QWizard.ModernStyle)

    # Customize button texts
    wizard.setButtonText(QWizard.BackButton, "Back")
    wizard.setButtonText(QWizard.NextButton, "Next")

    # Apply custom style
    wizard.button(QWizard.NextButton).setStyleSheet("""
        QPushButton {
            background-color: #21AE62;
            color: white;
            font-family: Helvetica, Arial;
            font-weight: bold;
            border-radius: 10px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #1E9955;
        }
        QPushButton:pressed {
            background-color: #187D45;
        }
        QPushButton:disabled {
            background-color: #A0D5BA;
            color: white;
        }
    """)

    wizard.button(QWizard.BackButton).setStyleSheet("""
        QPushButton {
            background-color: #CCCCCC;
            color: black;
            font-family: Helvetica, Arial;
            border-radius: 10px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #BBBBBB;
        }
        QPushButton:pressed {
            background-color: #AAAAAA;
        }
        QPushButton:disabled {
            background-color: #EEEEEE;
            color: #888888;
        }
    """)
    
    wizard.setWindowTitle("NeuroStemVolt")
    
    icon_path = os.path.join(base, "ui", "ui/graphics/NSV_Logo_Icon.png")
    app.setWindowIcon(QIcon(icon_path))
    wizard.addPage(IntroPage())
    wizard.addPage(ColorPlotPage())
    wizard.addPage(ResultsPage())
    
    def show_wizard():
        """Function to properly transition from splash to wizard"""
        splash.finish(wizard)  # This ensures splash closes when wizard shows
        wizard.show()
        wizard.raise_()  # Bring to front
        wizard.activateWindow()  # Give focus
    
    # Set up timer to show wizard after splash duration
    timer = QTimer()
    timer.timeout.connect(show_wizard)
    timer.setSingleShot(True)
    timer.start(3000)
    
    # Start the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()