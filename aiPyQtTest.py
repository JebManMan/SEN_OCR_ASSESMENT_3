import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QVBoxLayout, QWidget, QLabel)
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QGradient

class AnimatedButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 15px;
                border-radius: 15px;
                background-color: #2c3e50;
                color: white;
                border: 2px solid #34495e;
            }
            QPushButton:hover {
                background-color: #34495e;
                border: 2px solid #3498db;
            }
        """)
        
        # Store original size
        self.original_geometry = None

        # Hover animation
        self.hover_anim = QPropertyAnimation(self, b"geometry")
        self.hover_anim.setDuration(150)
        self.hover_anim.setEasingCurve(QEasingCurve.Type.OutQuad)

        # Click animation
        self.click_anim = QPropertyAnimation(self, b"geometry")
        self.click_anim.setDuration(400)
        self.click_anim.setEasingCurve(QEasingCurve.Type.OutBounce)

    def resizeEvent(self, event):
        if self.original_geometry is None:
            self.original_geometry = self.geometry()
        super().resizeEvent(event)

    def enterEvent(self, event):
        if self.original_geometry:
            self.hover_anim.setStartValue(self.geometry())
            self.hover_anim.setEndValue(QRect(
                self.original_geometry.x() - 3,
                self.original_geometry.y() - 3,
                self.original_geometry.width() + 6,
                self.original_geometry.height() + 6
            ))
            self.hover_anim.start()

    def leaveEvent(self, event):
        if self.original_geometry:
            self.hover_anim.setStartValue(self.geometry())
            self.hover_anim.setEndValue(self.original_geometry)
            self.hover_anim.start()

    def mousePressEvent(self, event):
        if self.original_geometry:
            self.click_anim.setStartValue(self.geometry())
            self.click_anim.setEndValue(QRect(
                self.original_geometry.x() - 6,
                self.original_geometry.y() - 6,
                self.original_geometry.width() + 12,
                self.original_geometry.height() + 12
            ))
            self.click_anim.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.original_geometry:
            self.click_anim.setStartValue(self.geometry())
            self.click_anim.setEndValue(self.original_geometry)
            self.click_anim.start()
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Training Interface")
        self.setMinimumSize(600, 400)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Add title
        title = QLabel("AI Model Interface")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)

        # Create buttons
        self.train_button = AnimatedButton("Training")
        self.test_button = AnimatedButton("Testing")

        # Add buttons to layout
        layout.addWidget(self.train_button)
        layout.addWidget(self.test_button)

        # Connect button signals
        self.train_button.clicked.connect(self.on_train_clicked)
        self.test_button.clicked.connect(self.on_test_clicked)

        # Set window background
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #f5f6fa,
                    stop: 1 #dcdde1
                );
            }
        """)

    def on_train_clicked(self):
        print("Training button clicked")
        # Add your training logic here

    def on_test_clicked(self):
        print("Testing button clicked")
        # Add your testing logic here

def generateGUT():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    generateGUT();