# This script: diffusion_canvas.py - Main script

# Overview of all scripts in project:
# scripts/diffusion_canvas.py - Script that interfaces with sd.webui and is the entry point for launch.
# brushes.py - Tools for image data manipulation.
# sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we might do to get
#                        things we need from sd.webui.
# shader_runner.py - Used to execute shader-based math on tensors.
# texture_convert.py - Automatic conversion of various representations of texture data.
# ui.py - UI for DiffusionCanvas.

import modules.scripts as scripts
from PyQt6.QtWidgets import QApplication
import gradio
import sys
from ui import DiffusionCanvasWindow
from sdwebui_interface import begin_interrupt, end_interrupt, unfreeze_sd_webui


class Script(scripts.Script):
    def __init__(self):
        pass

    def title(self):
        return "Diffusion Canvas"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        button = gradio.Button("Launch")
        button.click(fn=launch)
        return []

    def process(self, p):
        global running
        if not running:
            return

        p.batch_size = 1
        p.batch_count = 1


running: bool = False


def launch():
    # If it's already running, don't launch another one.
    global running
    if running:
        return

    print("Starting diffusion canvas...")
    running = True
    begin_interrupt()

    app = QApplication(sys.argv)
    window = DiffusionCanvasWindow()
    window.show()
    app.exec()

    end_interrupt()
    running = False
    print("...Diffusion canvas ended")