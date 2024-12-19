# This script: brushes.py - Tools for image data manipulation.

# Overview of all scripts in project:
# scripts/diffusion_canvas.py - Script that interfaces with sd.webui and is the entry point for launch.
# brushes.py - Tools for image data manipulation.
# sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we might do to get
#                        things we need from sd.webui.
# shader_runner.py - Used to execute shader-based math on tensors.
# texture_convert.py - Automatic conversion of various representations of texture data.
# ui.py - UI for DiffusionCanvas.
# diffusion_canvas_api.py - Contains functions used by the UI

import utils.texture_convert as conv
from shader_runner import ShaderRunner, Program, safe_release

sr: ShaderRunner | None = None
dab: Program | None = None

draw_dab_fragment_shader = '''
#version 330

uniform sampler2D InputImage;

// (1 / image size xy, image size xy)
uniform vec4 TexelSize;

uniform vec2 Center;
uniform float Radius;
uniform vec4 Color;
uniform float Opacity;
uniform int Mode;
in vec2 v_texcoord;
out vec4 fragColor;

#define MODE_BLEND 0
#define MODE_ADD 1

void main() {
    fragColor = texture(InputImage, v_texcoord);
    vec2 pixelCoord = v_texcoord * TexelSize.zw;
    float opacity = smoothstep(0.0, 1.0, 1.0 - length(pixelCoord - Center) / Radius) * Opacity;
    
    if(Mode == MODE_BLEND)
        fragColor = mix(fragColor, Color, opacity);
    else if(Mode == MODE_ADD)
        fragColor = fragColor + Color * opacity;
}
'''


class Brushes:
    modes = ("blend", "add")

    def __init__(self):
        self.sr = ShaderRunner()
        self.dab = self.sr.create_program(draw_dab_fragment_shader)

    def release(self):
        self.sr = safe_release(self.sr)

    def draw_dab(self,
                 image: conv.supported_types,
                 center: tuple[float, float],
                 radius: float,
                 color: tuple[float, float, float, float],
                 opacity: float = 1,
                 mode: str = "blend") \
            -> conv.supported_types:

        # Create texture from input image
        input_texture = conv.convert_to_moderngl_texture(image, self.sr.create_texture)
        # Create an empty texture for output
        output_texture = self.sr.create_empty_texture(input_texture.width, input_texture.height,
                                                      input_texture.components,
                                                      dtype=input_texture.dtype)

        self.dab.program["Center"] = center
        self.dab.program["Radius"] = radius
        self.dab.program["Color"] = color
        self.dab.program["Opacity"] = opacity
        self.dab.program["Mode"] = self.modes.index(mode) if mode in self.modes else 0

        self.dab.program["TexelSize"] = (
            1 / input_texture.width,
            1 / input_texture.height,
            input_texture.width,
            input_texture.height
        )

        # Bind the input texture to the shader
        self.dab.bind_textures([
            (input_texture, 'InputImage'),
        ])

        self.dab.run(output_texture)

        output = conv.convert(output_texture, type(image))
        self.sr.release_item(input_texture)
        self.sr.release_item(output_texture)
        return output
