# ------------------------------------------------------------
# QR Code Generator & Decoder
# ------------------------------------------------------------
# Author: Priyadharshni G
# Description:
#   This program allows users to generate and decode QR codes
#   through a graphical interface built with Tkinter.
#   - Generates QR codes from text or URLs and saves them as .png
#   - Decodes QR codes from image files and displays extracted data
# ------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox
import qrcode
import cv2
import os
from PIL import Image, ImageTk


# ------------------------------------------------------------
# Function: generate_qr
# Purpose: Generates a QR code from user-provided text or URL.
# ------------------------------------------------------------
def generate_qr():
    data = data_entry.get().strip()         # Fetch input text or URL
    filename = filename_entry.get().strip() # Fetch optional filename
    
    # Validate user input
    if not data:
        messagebox.showwarning("Input Error", "Please enter text or URL to generate QR code.")
        return
    
    # Default filename if user leaves field empty
    if not filename:
        filename = "default_qr.png"
    if not filename.endswith(".png"):
        filename += ".png"

    # Create QR code with high error correction
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Convert QR object to image and save as .png
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    
    # Notify user and display generated QR image in GUI
    messagebox.showinfo("Success", f"QR Code saved as {filename}")
    display_qr_image(filename)
  
    # Clear input fields after generation
    data_entry.delete(0, tk.END)
    filename_entry.delete(0, tk.END)


# ------------------------------------------------------------
# Function: decode_qr
# Purpose: Allows user to select and decode a QR image file.
# ------------------------------------------------------------
def decode_qr():
    # Open file dialog for user to select QR image
    filename = filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
    if not filename:
        return

    # Validate file existence
    if not os.path.exists(filename):
        messagebox.showerror("File Error", "File not found. Please select a valid file.")
        return
    
    # Load image and decode using OpenCV
    img = cv2.imread(filename)
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    
    # Display decoded text if found
    if data:
        messagebox.showinfo("Decoded Data", f"QR Code contains:\n{data}")
        
        # Copy decoded text to clipboard for convenience
        root.clipboard_clear()
        root.clipboard_append(data)
        root.update()
    else:
        messagebox.showinfo("Result", "No QR code found in the image.")


# ------------------------------------------------------------
# Function: display_qr_image
# Purpose: Displays the generated QR image in the GUI.
# ------------------------------------------------------------
def display_qr_image(filename):
    img = Image.open(filename)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    qr_image_label.config(image=img_tk)
    qr_image_label.image = img_tk


# ------------------------------------------------------------
# GUI SETUP
# ------------------------------------------------------------
root = tk.Tk()
root.title("QR Code Generator & Decoder")
root.geometry("450x500")
root.resizable(False, False)

# Input label and field for text/URL
tk.Label(root, text="Enter Text/URL:", font=("Helvetica", 12)).pack(pady=10)
data_entry = tk.Entry(root, font=("Helvetica", 12), width=40)
data_entry.pack()

# Input label and field for custom filename
tk.Label(root, text="Enter Filename (optional):", font=("Helvetica", 12)).pack(pady=10)
filename_entry = tk.Entry(root, font=("Helvetica", 12), width=40)
filename_entry.pack()

# Buttons for generating and decoding QR codes
tk.Button(root, text="Generate QR Code", font=("Helvetica", 12), bg="green", fg="white", command=generate_qr).pack(pady=15)
tk.Button(root, text="Decode QR Code", font=("Helvetica", 12), bg="blue", fg="white", command=decode_qr).pack(pady=10)

# Label placeholder to display QR image
qr_image_label = tk.Label(root)
qr_image_label.pack(pady=20)

# Run main event loop
root.mainloop()
