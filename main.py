import multiprocessing as mp
import tkinter as tk
from pathlib import Path
from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk
from transformers import AutoModelForCausalLM, AutoProcessor


def worker(q_in, q_out):
    checkpoint = "microsoft/git-base"
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    while True:
        if q_in.empty():
            continue

        img_file = q_in.get()
        inputs = processor(images=Image.open(img_file), return_tensors="pt")
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        q_out.put({"img_stem": img_file.stem, "cap": generated_caption})


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.directory = None
        self.all_files = None  # list of paths
        self.image_files = None  # list of paths
        self.current_image = None  # Path to image file
        self.current_caption = None  # Path to file with caption

        self.create_widgets()

        self.q_in = mp.Queue()
        self.q_out = mp.Queue()

        self.captioner_proc = mp.Process(target=worker, args=(self.q_in, self.q_out))
        self.captioner_proc.start()
        self.check_process_status()

    def create_widgets(self):
        self.master.geometry("600x600")
        self.paddings = {"padx": 5, "pady": 5}

        self.aside = tk.Frame(self.master)
        self.aside.pack(side=tk.LEFT, fill=tk.Y)

        self.main = tk.Frame(self.master)
        self.main.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.main)
        self.canvas.pack(expand=True, fill=tk.BOTH, **self.paddings)

        self.caption_frame = tk.Frame(self.main)
        self.caption_frame.pack(fill=tk.X)

        self.select_dir_btn = tk.Button(
            self.aside, command=self.select_dir, text="select directory"
        )
        self.select_dir_btn.pack(**self.paddings)

        self.listbox = tk.Listbox(self.aside)
        self.listbox.bind("<<ListboxSelect>>", self.show_selected_image)
        self.listbox.pack(expand=True, fill=tk.Y, **self.paddings)

        self.string_var = tk.StringVar(self.caption_frame, "")
        self.string_var.trace_add("write", self.update_caption)

        self.caption = tk.Entry(self.caption_frame, textvariable=self.string_var)
        self.caption.pack(side=tk.LEFT, expand=True, fill=tk.X, **self.paddings)

        self.generate_cap_btn = tk.Button(
            self.caption_frame, command=self.generate_caption, text="generate caption"
        )
        self.generate_cap_btn.pack(side=tk.LEFT, **self.paddings)

    def update_caption(self, *args):
        self.current_caption.write_text(self.string_var.get())

    def select_dir(self):
        supported_exts = (".png", ".jpg", ".jpeg", ".bmp")
        self.directory = filedialog.askdirectory(title="select directory")
        # images and caption files (.txt)
        self.all_files = [
            x
            for x in Path(self.directory).iterdir()
            if x.is_file() and str(x).lower().endswith((*supported_exts, ".txt"))
        ]
        self.image_files = [
            x for x in self.all_files if str(x).lower().endswith(supported_exts)
        ]

        # list only images
        self.listbox.delete(0, self.listbox.size())
        for image_file in self.image_files:
            self.listbox.insert(tk.END, image_file.stem)

    def resize_and_keep_aspect_ratio(self, img, max_width, max_height):
        width, height = img.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return resized_img

    def check_process_status(self):
        if not self.q_out.empty():
            res = self.q_out.get()
            # print(f"res: {res}")
            cap_file = Path(self.directory / Path(res["img_stem"] + ".txt"))
            if cap_file == self.current_caption:
                self.string_var.set(res["cap"])
            else:
                cap_file.write_text(res["cap"])
            self.all_files.append(cap_file)

        self.master.after(100, self.check_process_status)
        # print(".", end="")

    def show_selected_image(self, event):
        if not self.listbox.curselection():
            return  # when listbox clicked but none item was selected

        if self.current_image:  # delete old image
            self.canvas.delete("all")

        selected_index = self.listbox.curselection()[0]
        self.current_image = self.image_files[selected_index]

        img = Image.open(self.current_image)
        resized_img = self.resize_and_keep_aspect_ratio(img, 512, 512)
        photo = ImageTk.PhotoImage(resized_img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        # find caption .txt file and set it in text field
        self.current_caption = Path(
            self.directory / Path(self.current_image.stem + ".txt")
        )
        if self.current_caption in self.all_files:
            caption = self.current_caption.read_text()
            self.string_var.set(caption)
        else:
            self.string_var.set("")

    def generate_caption(self, *args):
        # print(f"req: {self.current_image.stem}")
        self.q_in.put(self.current_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
