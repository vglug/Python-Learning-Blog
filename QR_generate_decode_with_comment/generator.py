import qrcode

def create_qr(data, filename):
    # Create a QR code object
    qr = qrcode.QRCode(
        version=1,
        box_size=10,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Create the image
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    print(f"QR Code saved as {filename}")

# Example usage
if __name__ == "__main__":
    file_name=input("Enter your file name :")
    data = input("Enter text or URL to generate QR code: ")
    create_qr(data, file_name)
