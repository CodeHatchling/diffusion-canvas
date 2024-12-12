# diffusion-canvas
Extension for sd.webui that offers a brush-style interactive denoising interface

HOW TO INSTALL:

1. Unzip the contents of "replace_files_in_modules.zip" into "sd.webui\webui\modules", replacing the files.
   - Note: sd.webui version tested is 1.10.1
2. Install the extension normally. Files should appear in "sd.webui\webui\extensions".
   - If necessary, simply download the contents into "sd.webui\webui\extensions\diffusion-canvas"

HOW TO USE:

1. Open sd.webui
2. Press [LAUNCH] button in extensions.
3. LEFT CLICK and drag anywhere on the canvas to add noise.
4. Remove noise by...
   - Entering your prompt, loras, CFG, etc. in sd.webui.
   - Press [GENERATE]. sd.webui will get stuck, this is intended - at this point, sd.webui has passed the network and settings to diffusion canvas.
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
   - Denoise Attenuation: The proportion of noise removed from the image for each denoise click. For example, 0.2 will remove 20% of the noise each time.
   - Denoise Subtraction: The absolute amount of noise subtracted. For example, if an image has a noise level of 0.5, a subtraction value of 0.1 will reduce it to 0.4.
     - Both attenuation and subtraction occurs with each right "denoising brush stroke".
     - The lower the attenuation and subtraction, the more steps it takes to resolve the image, and usually results in better image quality.
     - Recommended settings for quality: attenuation=0.05, subtraction=0.01
     - Recommended settings for rough draft: attenuation=0.5, subtraction=0.01
     - Recommended settings for general work: attenuation=0.2, subtraction=0.01
7. Use the [New] [Save] and [Load] buttons on the top left.
   - When saving, remember to use the correct extension for the file type (e.g. ".png")
