#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

def main():
    app = QApplication(sys.argv)
    
    # Create a simple window with a button
    window = QMainWindow()
    window.setWindowTitle("Test Window")
    window.setGeometry(100, 100, 400, 300)
    
    # Create central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    # Create layout
    layout = QVBoxLayout(central_widget)
    
    # Create a button
    button = QPushButton("Test Button")
    button.clicked.connect(lambda: print("Button clicked!"))
    layout.addWidget(button)
    
    # Show the window
    window.show()
    
    print("Window should be visible now")
    
    # Start the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 