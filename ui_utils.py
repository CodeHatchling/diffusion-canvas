from PyQt6.QtWidgets import QWidget, QMessageBox


class ExceptionCatcher:
    def __init__(self, context: QWidget | None, message: str):
        self.context = context
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            import traceback as tb
            except_string = '\n'.join(tb.format_exception(exc_type, exc_value, traceback))
            QMessageBox.critical(self.context, f"Error: {exc_type}", f"{self.message}: {exc_value}\n"
                                                                     f"\n" +
                                                                     except_string)
        return True  # Suppress exceptions