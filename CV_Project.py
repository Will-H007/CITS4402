import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from exceptiongroup import catch
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import tables as tb
import io


class ImageGUI:
   
    def __init__(self, master):
        self.master = master
        self.master.title("CV Project")
       
        # Create a frame for the GUI and center it
        self.frame = tk.Frame(self.master)
        self.frame.pack(expand=True, padx=10, pady=10)
        self.frame.grid_rowconfigure(0,  weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Create a border for the Controls
        self.border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.border.grid(row=0, column=0, sticky="nsew")

        # Create a border for the ImageFrame
        self.image_border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.image_border.grid(row=0, column=1, sticky="nsew")

        # Create a label to display the chosen image
        self.image_label = tk.Label(self.image_border)
        self.image_label.grid(column = 1, row = 1, sticky="nsew", padx = 10, pady = 10)

        # Create a "Load Slice Directory" button
        self.load_button = tk.Button(self.border, text="Load Slice Directory", command=self.load_directory)
        self.load_button.grid(column = 1, row = 1, padx = 10, pady = 10, sticky="w")

        # Create a Label for channel OptionMenu
        self.channel_label = tk.Label(self.border, text=' Channel ', relief=tk.FLAT)
        self.channel_label.grid(column = 1, row = 2, padx = 10, pady = 10, sticky="sw")

        # Create a Option menu for setting channel
        self.channel_var = tk.StringVar()
        self.channel_var.set('T1')
        self.channel_option = tk.OptionMenu(self.border, self.channel_var, "T1", "T1Gd", "T2", "T2 FLAIR", command=self.updateValue)
        self.channel_option.config(width=10)
        self.channel_option.grid(column = 1, row = 3, padx = 10, pady = 10, sticky="nsew")

        # Create a Label for Annotation OptionMenu
        self.channel_label = tk.Label(self.border, text=' Annotation ', relief=tk.FLAT)
        self.channel_label.grid(column = 1, row = 4, padx = 10, pady = 10, sticky="sw")

        # Create a Option menu for setting annotation
        self.annotation_var = tk.StringVar()
        self.annotation_var.set('Off')
        self.annotation_option = tk.OptionMenu(self.border, self.annotation_var, "Off", "On", command=self.updateValue)
        self.annotation_option.config(width=10)
        self.annotation_option.grid(column = 1, row = 5, padx = 10, pady = 10, sticky="nsew")

        # Create a "Extract Conventional Features" button
        self.conventional_button = tk.Button(self.border, text="Extract Conventional Features", command=self.updateValue)
        self.conventional_button.grid(column = 1, row = 6, padx = 10, pady = 10)

        # Create a "Extract Radiomic Features" button
        self.radiomic_button = tk.Button(self.border, text="Extract Radiomic Features", command=self.updateValue)
        self.radiomic_button.grid(column = 1, row = 7, padx = 10, pady = 10, sticky="w")

    def load_directory(self):
        # Open a file selection dialog box to choose an image file
        self.file_path = filedialog.askdirectory(title="Select Volume Folder")
        print(self.file_path)
        self.showSliceImage(0)
        self.showSliceIDBar()

    def showSliceImage(self, index):
        entries = os.listdir(self.file_path+'/')
        print(entries)
        volume = entries[0].split('_')[1]
        file_name = 'volume_{}_slice_{}.h5'.format(volume, str(index))
        full_path = self.file_path + '/' + file_name
        print(full_path)
        file = tb.open_file(full_path, mode='r')
        print(file.root.image)
        
        # Get Image
        glioma_images = file.root.image.read()
        # Get Mask
        mask_images = file.root.mask.read()
        file.close()

        # merge non-overlapping masks by addition
        merge_one = mask_images[:,:,0] + mask_images[:,:,1] + mask_images[:,:,2]
        print('Mask Shape: ', merge_one.shape)
        
        # Get Channel index
        channels = {"T1": 0, "T1Gd": 1, "T2": 2, "T2 FLAIR": 3}
        channel_index = channels.get(self.channel_var.get())
        print('channel_index: ' + str(channel_index))
        #print(glioma_images[:,:,channel_index])
        print(glioma_images.shape)
        
        channel_image = glioma_images[:,:,channel_index]
        print(channel_image.shape)

        image_file = io.BytesIO()
        mask_file = io.BytesIO()
        plt.imsave(image_file, channel_image, cmap = 'gray')
        plt.imsave(mask_file, merge_one, cmap = 'gray')
        
        print(image_file)
        pil_image = Image.open(image_file)
        pil_mask = Image.open(mask_file)

        annotation = self.annotation_var.get() == 'On'
        
        print('annotation: ', annotation)
        if (annotation):
            pil_image = Image.blend(pil_image, pil_mask, 0.5)

        # Resize the image to fit in the image_label label
        width, height = pil_image.size
        print(width,height)
        photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        

    def showSliceIDBar(self):
        #  Create a Scale widget for setting Slice ID
        self.slice_var = tk.IntVar()
        self.slice_var.set(0)
        self.slice_scale = tk.Scale(self.image_border, width=20, length = 310, from_=0, to=154, orient=tk.HORIZONTAL, label="Slice ID", variable=self.slice_var)
        self.slice_scale.bind("<ButtonRelease-1>", self.updateValue)
        self.slice_scale.grid(column = 1, row = 2, sticky="sw", padx = 10, pady = 10)
  

    def updateValue(self, event):
        evt_name = str(event)
        print(evt_name)
        slice_id = self.slice_var.get()
        print('slice_id: ', slice_id)
        self.showSliceImage(slice_id)

 
            
if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()




