from PyQt6.QtWidgets import (
    QLabel,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QSpinBox
)

from ui_utils import ExceptionCatcher


class NewCanvasDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Canvas")

        # Layout for the dialog
        self.layout = QGridLayout(self)

        # Width input
        self.layout.addWidget(QLabel("Width:"), 0, 0)
        self.width_input = QSpinBox(self)
        self.width_input.setRange(8, 8192)  # Range for width
        self.width_input.setValue(512)      # Default value
        self.layout.addWidget(self.width_input, 0, 1)

        # Height input
        self.layout.addWidget(QLabel("Height:"), 1, 0)
        self.height_input = QSpinBox(self)
        self.height_input.setRange(8, 8192)  # Range for height
        self.height_input.setValue(512)      # Default value
        self.layout.addWidget(self.height_input, 1, 1)

        # Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons, 2, 0, 1, 2)

    def get_dimensions(self):
        with ExceptionCatcher(self, "Failed to get dimensions"):
            return self.width_input.value(), self.height_input.value()
