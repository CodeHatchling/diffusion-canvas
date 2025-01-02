# diffusion-canvas
Extension for sd.webui that offers a brush-style interactive denoising interface

HOW TO INSTALL:

1. Unzip the contents of "replace_files_in_modules.zip" into "sd.webui\webui\modules", replacing the files.
   - Note: sd.webui version tested is 1.10.1
2. Install the extension normally. Files should appear in "sd.webui\webui\extensions".
   - If necessary, simply download the contents into "sd.webui\webui\extensions\diffusion-canvas"
3. Install this stuff into your python environment (I'd explain how, but pycharm does this for me automatically and I hate consoles):
   - moderngl
   - PyQt6

HOW TO USE:

1. Open sd.webui
2. Press [LAUNCH] button in extensions.
3. LEFT CLICK and drag anywhere on the canvas to add noise.
4. Remove noise by...
   - Entering your prompt, LoRAs, CFG, etc. in sd.webui.
   - Press [GENERATE]. sd.webui will get stuck, this is intended - at this point, sd.webui has passed the network and settings to Diffusion Canvas.
   - A button will appear in Diffusion Canvas' "Params Palette" dock. Click it to change to this prompt/setting at any time.
   - A "param" must be selected before you can denoise.
   - RIGHT CLICK and drag anywhere on the canvas to remove noise around the mouse cursor.
5. Add new prompts by...
   - Pressing [Unfreeze sd.webui] on top of Diffusion Canvas.
   - Switch back to sd.webui to enter a new prompt/settings/etc.
   - Press [GENERATE] to pass these new settings to Diffusion Canvas.
   - A new button will appear in the "Params Palette". Click this button to use your new settings.
6. Change settings...
   - Noise Radius (px): The radius of the noise brush in pixels. You can enter any value in the numeric entry field.
   - Noise Intensity: The amount of noise to add when you click. If you add too much the canvas may turn black; simply denoise to fix this.
   - Context Width/Height (px): The size of region sent to the denoiser. The network will denoise anything in this region (and can only see things in this region).
     - The context size is rounded to a multiple of 8 because latents are 8x8 pixels each.
   - Denoise Attenuation: The proportion of noise removed from the image for each denoise click. For example, 0.2 will remove 20% of the noise each time.
   - Denoise Subtraction: The absolute amount of noise subtracted. For example, if an image has a noise level of 0.5, a subtraction value of 0.1 will reduce it to 0.4.
     - Both attenuation and subtraction occurs with each denoising brush stroke.
     - The lower the attenuation and subtraction, the more steps it takes to resolve the image, and usually results in better image quality.
     - Recommended settings for quality: attenuation=0.05, subtraction=0.01
     - Recommended settings for rough draft: attenuation=0.5, subtraction=0.01
     - Recommended settings for general work: attenuation=0.2, subtraction=0.01
   - Denoise Bias: Controls how aggressive or passive the denoiser should be. Similar in concept to "temperature" for LLMs.
     - Negative values: Remove less noise. This will tend to add more detail to the image, but can harm coherence and clarity.
     - Positive values: Remove more noise. This will make the image smoother and more coherent, but tends to produce blurrier images.
     - Usage tips:
       - For areas that need more detail, selectively denoise with a slight negative bias.
       - For areas that lack clarity, use a slight positive bias.
       - By changing the bias at different points in the denoising process, you can add/remove detail/coherence at different "scales".
       - If you use a positive bias at higher noise levels, large scale structure may be more coherent.
       - If you use a negative bias at low noise levels, more fine detail may be added.
7. Use the [New] [Save] and [Load] buttons on the top left.
   - When saving, if you do not enter an extension (e.g. ".png") it will save it as a PNG file.

NOTES: 
- If you use Diffusion Canvas while sd.webui is NOT frozen, some features (like LoRAs) will be deactivated.
- Diffusion Canvas will use the LoRA settings provided by the most recent call to [GENERATE]. To change LoRAs, follow step 5.

CREDIT:
- This software uses a procedure similar to the inpainting technique described in "Differential Diffusion".

PLANNED FEATURES/FIXES:

- Textured Brushes:
  - History/undo brush. Instead of a solid color, provide a previous version of the image as content to paint in.
  - Clone/stamp brush. The last undo state is used as a source, but is painted with an offset.
  - Texture brush. Upload an image that can be used as a source, options for tiling and random offsets.
  - Custom: Any combination of the above features; source latents, offset, tiling, etc.
  - Deghosting. When painting in a texture, the transition zones are noised and denoised.
- A "denoise all" checkbox for the noise/denoise brush. (Work-around: enter a large value like 99999.)
- Pages for color swatches. Currently they all pile up under the color picker without limit. 
- Branched undo history.
- Save/load encoded latent data. This can be used to avoid the loss that decoding and encoding causes.
- Save/load params palette.
- Generate on the main (sd.webui) thread instead of the UI thread.
