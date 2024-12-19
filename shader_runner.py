# This script: shader_runner.py - Used to execute shader-based math on tensors.

# Overview of all scripts in project:
# scripts/diffusion_canvas.py - Script that interfaces with sd.webui and is the entry point for launch.
# brushes.py - Tools for image data manipulation.
# sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we might do to get
#                        things we need from sd.webui.
# shader_runner.py - Used to execute shader-based math on tensors.
# texture_convert.py - Automatic conversion of various representations of texture data.
# ui.py - UI for DiffusionCanvas.
# diffusion_canvas_api.py - Contains functions used by the UI

import PIL.Image
import numpy as np
import moderngl


def can_release(obj):
    if obj is None:
        return False
    if not hasattr(obj, "release"):
        return False
    return callable(obj.release)


def safe_release(obj):
    if can_release(obj):
        obj.release()
        return None
    return obj


class Program:
    def __init__(self, parent: 'ShaderRunner', fragment_shader: str):
        """
        Compiles a runnable program.

        Parameters:
        - context (Context): The context.
        - vbo (Buffer): The full screen quad.
        - vertex_shader (str): The GLSL source code for the full screen quad.
        - fragment_shader (str): The GLSL source code for the fragment shader.
        """
        self.parent = parent
        self.program = parent.ctx.program(vertex_shader=parent.vertex_shader, fragment_shader=fragment_shader)
        self.vao = parent.ctx.simple_vertex_array(self.program, parent.vbo, 'in_position', 'in_texcoord')

    def release(self):
        self.vao = safe_release(self.vao)
        self.program = safe_release(self.program)

    def bind_textures(self, textures):
        # TODO: Resolve conflicts that may occur with a texture shared across multiple programs.
        for i, (texture, name) in enumerate(textures):
            texture.use(location=i)
            self.program[name].value = i

    def run(self, output_texture: moderngl.Texture):
        """
        Run the shader program and render to the given output texture.

        Parameters:
        - output_texture (moderngl.Texture): The texture to render output to.
        """
        fbo = self.parent.get_frame_buffer(output_texture)

        # Bind the framebuffer and render
        fbo.use()
        self.parent.ctx.clear(0.0, 0.0, 0.0, 0.0)  # Clear the framebuffer
        self.vao.render(moderngl.TRIANGLE_STRIP)


class ShaderRunner:
    def __init__(self):
        """
        Initialize the ShaderRunner.
        """
        # Create OpenGL context
        self.ctx = moderngl.create_standalone_context()

        self.all_textures = []      # List to hold all textures for cleanup
        self.all_programs = []      # List to hold all programs for cleanup

        # Vertex shader source code
        self.vertex_shader = '''
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            v_texcoord = in_texcoord;
        }
        '''

        # Set up vertex array object (full-screen quad)
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        # Holds onto the current state of the frame buffer.
        self.fbo = None
        self.output_texture = None

    def create_program(self, fragment_shader: str) -> Program:
        program = Program(self, fragment_shader)
        self.all_programs.append(program)
        return program

    def __enter__(self):
        # Return self to allow access to methods within the 'with' block
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Release resources
        self.release()
        # Propagate exceptions (if any)
        return False

    def create_texture(self, size: tuple[int, int], channels: int, data, dtype) -> moderngl.Texture:
        tex_gpu = self.ctx.texture(size=size, components=channels, data=data, dtype=dtype)
        self.all_textures.append(tex_gpu)
        return tex_gpu

    def get_frame_buffer(self, output_texture: moderngl.Texture) -> moderngl.Framebuffer:
        if output_texture is not self.output_texture:
            # If the output texture has changed, recreate the framebuffer
            if self.fbo is not None:
                self.fbo.release()
            self.fbo = self.ctx.framebuffer(color_attachments=[output_texture])
            self.output_texture = output_texture

        return self.fbo

    def create_empty_texture(self, width, height, channels, dtype='f1') -> moderngl.Texture:
        """
        Create an empty texture on the GPU.

        Parameters:
        - width (int): Width of the texture.
        - height (int): Height of the texture.
        - channels (int): Number of channels.
        - dtype (str): Data type of the texture (default 'f1' for float32).

        Returns:
        - tex_gpu (moderngl.Texture): The created GPU texture.
        """
        tex_gpu = self.ctx.texture((width, height), channels, dtype=dtype)
        self.all_textures.append(tex_gpu)
        return tex_gpu

    def release_item(self, item):
        if isinstance(item, moderngl.Texture):
            if item in self.all_textures:
                safe_release(item)
                self.all_textures.remove(item)

        if isinstance(item, Program):
            if item in self.all_programs:
                safe_release(item)
                self.all_programs.remove(item)

    def release(self):
        # Release the resources in reverse order
        for tex_gpu in self.all_textures:
            if can_release(tex_gpu):
                tex_gpu.release()
        self.all_textures = []

        for program in self.all_programs:
            if can_release(program):
                program.release()
        self.all_programs = []

        self.fbo = safe_release(self.fbo)
        self.output_texture = None
        self.vbo = safe_release(self.vbo)
        self.vertex_shader = None
        self.ctx = safe_release(self.ctx)


def func():
    # Fragment shader that inverts the colors
    fragment_shader = '''
    #version 330

    uniform sampler2D InputImage;
    in vec2 v_texcoord;
    out vec4 fragColor;

    void main() {
        // Sample the input image
        vec4 color = texture(InputImage, v_texcoord);
        // Invert the colors
        fragColor = vec4(1.0 - color.rgb, color.a);
    }
    '''

    from PIL import Image
    import utils.texture_convert as conv

    # Load the image
    input_image = Image.open("C:/Users/yoshi/OneDrive/Pictures/dragon flame club.png").convert('RGBA')

    with ShaderRunner() as sr:
        # Create texture from input image
        input_texture = conv.convert_to_moderngl_texture(input_image, sr.create_texture)
        # Create an empty texture for output
        output_texture = sr.create_empty_texture(input_image.width, input_image.height, 4, dtype='f1')

        program = sr.create_program(fragment_shader)

        # Bind the input texture to the shader
        program.bind_textures([
            (input_texture, 'InputImage'),
        ])

        program.run(output_texture)

        # Convert the output texture to a PIL image
        import utils.texture_convert as conv
        result_image = conv.convert(output_texture, PIL.Image.Image)
        result_image.show()


if __name__ == '__main__':
    func()
