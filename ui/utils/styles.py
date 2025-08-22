from PyQt5.QtWidgets import QPushButton, QListWidget

def apply_custom_styles(widget):
    """
    Apply custom visual styles to QPushButton or QListWidget widgets.

    This function assigns background color, font, padding, and hover effects
    depending on the button label content (e.g., "save", "clear") or widget type.

    Args:
        widget (QWidget): The widget to which styles should be applied. 
            Currently supports:
                - QPushButton: Styles vary by function (e.g., "save", "clear", general).
                - QListWidget: Styled with a pastel background and bold text.

    Returns:
        None
    """
    if isinstance(widget, QPushButton):
        label = widget.text().lower()

        if "save" in label or "export" in label:
            # Save/export buttons (blue)
            widget.setStyleSheet("""
                QPushButton {
                    background-color: #4178F2;
                    color: white;
                    font-family: Helvetica, Arial;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #3366CC;
                }
                QPushButton:pressed {
                    background-color: #264A9E;
                }
                QPushButton:disabled {
                    background-color: #AFCBF9;
                    color: white;
                }
            """)
        elif "clear" in label:
            # Clear buttons (pink/red)
            widget.setStyleSheet("""
                QPushButton {
                    background-color: #FF3877;
                    color: white;
                    font-family: Helvetica, Arial;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #E5306A;
                }
                QPushButton:pressed {
                    background-color: #C12459;
                }
                QPushButton:disabled {
                    background-color: #F5A9C4;
                    color: white;
                }
            """)
        else:
            # General buttons (green)
            widget.setStyleSheet("""
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
    
    elif isinstance(widget, QListWidget):
        widget.setStyleSheet("""
            QListWidget {
                background-color: #CAF2FB;
                color: black;
                font-family: Helvetica, Arial;
                font-weight: bold;
                border-radius: 8px;
                padding: 4px;
            }
        """)