# Description
Tiny tkinter application for image captioning with automatic caption generation option. After opening an image, the app creates a `.txt` file with the same name as the image file in the same directory as the image file.
Default model used for captioning is `microsoft/git-base` (hardcoded), but you can change one line and have others. Model inference on CPU with FP32 precision (slow).

# How to run:
1. clone repo
2. go to the folder
3. install packages
   - in conda: `$ conda create --name <env> --file env.txt`
   - with pip: `pip install transformers==4.36.2 pillow==10.0.1`
5. run the application `python3 main.py`.

# TODO:
- faster inference (quantized weights & GPU support)
- Ability to set different model
- nicer GUI
