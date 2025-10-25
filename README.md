# QR Code Generator & Decoder

A Python-based GUI application using Tkinter that allows users to generate and decode QR codes.  
This project demonstrates GUI programming, file handling, and integration with QR code libraries in Python.

## Features

- Generate QR codes from text or URLs.  
- Save QR codes as PNG image files.  
- Decode existing QR codes from image files.  
- Display generated QR codes within the GUI.  
- Automatically copy decoded data to the clipboard.  

## Example

Generating a QR code:  
`Input: https://example.com`  
`Output: QR code image saved as default_qr.png`

Decoding a QR code:  
`Select QR code image file`  
`Output: Decoded data displayed and copied to clipboard`

## Code Overview

The program includes three main functions:

1. **`generate_qr()`**  
   Generates a QR code from user input and saves it as a PNG file. Displays it in the GUI.

2. **`decode_qr()`**  
   Allows the user to select a QR code image, decodes it, and shows the content in a message box. Copies content to clipboard.

3. **`display_qr_image(filename)`**  
   Loads and resizes the generated QR code image for display in the GUI.

## File Information

- **File Name:** `QRCodeGeneratorDecoder.py`  
- **Author:** Priyadharshni G  

## Concepts Used

- Tkinter GUI design  
- QR code generation with `qrcode` library  
- Image handling with `Pillow` (PIL)  
- QR code decoding using `OpenCV`  
- File handling and clipboard integration  


