from pyzbar.pyzbar import decode
from PIL import Image

def decode_qr(filename):
    img = Image.open(filename)
    result = decode(img)

    if result:
        for qr in result:
            print("QR Code Data:", qr.data.decode('utf-8'))
    else:
        print("No QR code found in the image.")

# Example usage
if __name__ == "__main__":
    filename = input("Enter QR image file name: ")
    decode_qr(filename)
