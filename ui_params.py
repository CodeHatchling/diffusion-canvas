from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QDialog,
    QDialogButtonBox,
    QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

from typing import Callable

import torch
from ui_utils import ExceptionCatcher


class EditParamsDialog(QDialog):
    class Output:
        def __init__(self, name: str, pixmap: QImage | None):
            self.name: str = name
            self.pixmap: QPixmap | None = pixmap

    def __init__(self,
                 current_widget: "ParamsWidget",
                 generate_handler: Callable[[int, int, int, any], torch.Tensor] | None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Params")
        self.generate_handler = generate_handler

        self.delete = False

        self.layout = QFormLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add a name text field.
        self.name_input = QLineEdit(self)
        self.name_input.setText(current_widget.name)
        self.layout.addRow("Name", self.name_input)

        # Add a thumbnail preview.
        self.image = QLabel(self)
        self.image.setFixedWidth(200)
        self.image.setFixedHeight(200)
        self.image.setScaledContents(True)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if isinstance(current_widget.pixmap, QPixmap):
            self.image.setPixmap(current_widget.pixmap)
        else:
            self.image.setText("No Thumbnail")
        self.layout.addRow("Thumbnail", self.image)

        # Add a generate button.
        self.generate_button = QPushButton("Generate Thumbnail")
        self.generate_button.clicked.connect(lambda: self.generate_thumbnail(current_widget.params))
        self.layout.addWidget(self.generate_button)

        # Add Okay/Cancel buttons.
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def generate_thumbnail(self, params):
        with ExceptionCatcher(None, "Failed to generate thumbnail"):
            if self.generate_handler is None:
                return

            if params is None:
                return

            import utils.texture_convert as conv
            image = conv.convert(self.generate_handler(512, 512, 20, params), QImage)
            image = QPixmap.fromImage(image)
            self.image.setPixmap(image)

    def get_output(self) -> Output:
        return EditParamsDialog.Output(
            name=self.name_input.text(),
            pixmap=self.image.pixmap(),
        )


class ParamsWidget(QWidget):

    params: any
    delete_handler: Callable[['ParamsWidget'], None]
    button_image: QLabel
    button_label: QLabel

    def __init__(self,
                 params: any,
                 button_name: str,
                 params_setter: Callable[[any], None],
                 delete_handler: Callable[['ParamsWidget'], None],
                 generate_handler: Callable[[int, int, int, any], torch.Tensor] | None,
                 parent: QWidget | None = None,):
        super().__init__(parent)
        self.params = params
        self.delete_handler = delete_handler
        self.generate_handler = generate_handler

        # Create a layout for the button and rename action
        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        def add_button() -> QWidget:
            # Params selection button
            button = QPushButton(self)
            button.clicked.connect(lambda: params_setter(self.params))
            button.setFixedWidth(100)
            button.setFixedHeight(120)

            # Internal button layout
            button_internal_layout = QVBoxLayout(button)
            button.setLayout(button_internal_layout)
            button_internal_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.button_image = QLabel()
            self.button_image.setFixedWidth(80)
            self.button_image.setFixedHeight(80)
            self.button_image.setScaledContents(True)
            self.button_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            button_internal_layout.addWidget(self.button_image)

            self.button_label = QLabel()
            self.button_label.setText(button_name)
            self.button_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            button_internal_layout.addWidget(self.button_label)

            return button

        def add_side_buttons() -> QWidget:
            # Must be a widget.
            holder = QWidget(self)

            # Vertical layout for rename and delete buttons
            small_button_layout = QVBoxLayout()
            small_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            small_button_layout.setSpacing(0)
            small_button_layout.setContentsMargins(0, 0, 0, 0)

            # Edit button
            rename_button = QPushButton("📝", self)
            rename_button.setFixedWidth(30)
            rename_button.setFixedHeight(30)
            rename_button.clicked.connect(self.on_rename_params)
            small_button_layout.addWidget(rename_button)

            # Delete button
            delete_button = QPushButton("❌", self)
            delete_button.setFixedWidth(30)
            delete_button.setFixedHeight(30)
            delete_button.clicked.connect(self.on_delete)
            small_button_layout.addWidget(delete_button)

            holder.setLayout(small_button_layout)
            return holder

        main_layout.addWidget(add_button())
        main_layout.addWidget(add_side_buttons())

        # Add to layout
        self.setLayout(main_layout)

    def on_rename_params(self):
        """
        Opens a dialog to rename a params button.
        """
        with ExceptionCatcher(None, "Failed to handle rename request"):
            dialog = EditParamsDialog(
                current_widget=self,
                generate_handler=self.generate_handler,
                parent=None
            )

            if dialog.exec() == QDialog.DialogCode.Accepted:
                output = dialog.get_output()
                self.button_image.setPixmap(output.pixmap)
                self.button_label.setText(output.name)

    def on_delete(self):
        with ExceptionCatcher(None, "Failed to handle delete request"):
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Delete Params")  # Set the title
            dialog.setText(f"Do you want to delete the \"{self.name}\" params?")  # Set the main message
            dialog.setInformativeText("Press \"Delete\" to delete them, press \"Cancel\" to keep them.\n"
                                      "This cannot be undone.")  # Optional detailed message
            dialog.setIcon(QMessageBox.Icon.Warning)  # Set an icon (e.g., Question, Information, Warning, Critical)

            # Add custom buttons
            accept_button = dialog.addButton("Delete", QMessageBox.ButtonRole.AcceptRole)
            dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            # Execute the dialog
            dialog.exec()

            # Determine which button was clicked
            if dialog.clickedButton() == accept_button:
                self.delete_handler(self)

    def _get_name(self) -> str:
        return self.button_label.text()

    def _set_name(self, value: str):
        with ExceptionCatcher(None, "Failed to assign name"):
            self.button_label.setText(value)

    name = property(fget=_get_name, fset=_set_name)

    def _get_pixmap(self) -> QPixmap:
        return self.button_image.pixmap()

    def _set_pixmap(self, value: QPixmap):
        with ExceptionCatcher(None, "Failed to assign pixmap"):
            self.button_image.setPixmap(value)

    pixmap = property(fget=_get_pixmap, fset=_set_pixmap)