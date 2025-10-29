import qrcode
from pyzbar.pyzbar import decode
from PIL import Image
import cv2

# ---------- CREATE QR CODE ----------
def create_qr(data, filename="my_qr.png"):
    qr = qrcode.QRCode(
        version=1,  # version: size of QR code (1 smallest)
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,  # size of each box
        border=4,     # border size
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    print(f"[+] QR code created and saved as {filename}")

# ---------- DECODE QR CODE FROM IMAGE ----------
def decode_qr_image(filename):
    img = Image.open(filename)
    result = decode(img)

    if result:
        for qr_code in result:
            data = qr_code.data.decode('utf-8')
            print(f"[+] Decoded data from image: {data}")
    else:
        print("[-] No QR code found in the image.")

# ---------- DECODE QR CODE USING CAMERA ----------
def decode_qr_camera():
    cap = cv2.VideoCapture(0)
    print("[*] Opening camera... (Press 'q' to quit)")

    while True:
        success, frame = cap.read()
        for code in decode(frame):
            data = code.data.decode('utf-8')
            print(f"[+] QR Code detected: {data}")
            # Draw a rectangle around QR code
            pts = code.polygon
            pts = [(p.x, p.y) for p in pts]
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (0, 255, 0), 3)

        cv2.imshow('QR Code Scanner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- MAIN FUNCTION ----------
if _name_ == "_main_":
    print("===== QR Code Generator & Decoder =====")
    print("1. Create QR Code")
    print("2. Decode from Image")
    print("3. Decode using Camera")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        data = input("Enter data/text to encode: ")
        filename = input("Enter filename (with .png extension): ")
        create_qr(data, filename)

    elif choice == "2":
        filename = input("Enter QR image filename to decode: ")
        decode_qr_image(filename)

    elif choice == "3":
        decode_qr_camera()

    else:
        print("Exiting....")
