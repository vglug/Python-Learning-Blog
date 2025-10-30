# Importing required libraries
import qrcode                     # For generating QR codes
from pyzbar.pyzbar import decode  # For decoding QR codes from images or camera
from PIL import Image             # For handling image files
import cv2                        # For accessing and processing camera frames

# ---------- CREATE QR CODE ----------
def create_qr(data, filename="my_qr.png"):
    """
    Function to create a QR code from input text or data.
    Saves the QR code as an image file (default name: my_qr.png).
    """
    # Create a QRCode object with configuration options
    qr = qrcode.QRCode(
        version=1,  # Version controls the size of the QR code (1 = smallest)
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,  # Size of each small box in the QR grid
        border=4,     # Width of the white border around QR code
    )

    # Add data to the QR code
    qr.add_data(data)
    qr.make(fit=True)  # Adjust size automatically based on data length

    # Create and save the QR code image
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    print(f"[+] QR code created and saved as {filename}")

# ---------- DECODE QR CODE FROM IMAGE ----------
def decode_qr_image(filename):
    """
    Function to decode a QR code from an image file.
    Displays the decoded data (text/link/etc).
    """
    # Open the image containing the QR code
    img = Image.open(filename)

    # Decode the QR code using pyzbar
    result = decode(img)

    # If decoding was successful, print the data
    if result:
        for qr_code in result:
            data = qr_code.data.decode('utf-8')  # Convert byte data to string
            print(f"[+] Decoded data from image: {data}")
    else:
        print("[-] No QR code found in the image.")

# ---------- DECODE QR CODE USING CAMERA ----------
def decode_qr_camera():
    """
    Function to decode QR codes live using the computer's webcam.
    Press 'q' to exit the camera window.
    """
    # Start capturing video from the default camera (index 0)
    cap = cv2.VideoCapture(0)
    print("[*] Opening camera... (Press 'q' to quit)")

    while True:
        success, frame = cap.read()  # Capture each frame from the camera

        # Detect and decode QR codes in the current frame
        for code in decode(frame):
            data = code.data.decode('utf-8')
            print(f"[+] QR Code detected: {data}")

            # Get the coordinates of the QR code polygon (border points)
            pts = code.polygon
            pts = [(p.x, p.y) for p in pts]

            # Draw a green rectangle around the detected QR code
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (0, 255, 0), 3)

        # Display the video feed with any detected QR codes outlined
        cv2.imshow('QR Code Scanner', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close window after exiting
    cap.release()
    cv2.destroyAllWindows()

# ---------- MAIN FUNCTION ----------
if __name__ == "__main__":  # Correct Python syntax (__name__ not _name_)
    print("===== QR Code Generator & Decoder =====")
    print("1. Create QR Code")
    print("2. Decode from Image")
    print("3. Decode using Camera")
    print("4. Exit")

    # Ask user to choose an option
    choice = input("Enter your choice: ")

    # Option 1: Create QR code from user input
    if choice == "1":
        data = input("Enter data/text to encode: ")
        filename = input("Enter filename (with .png extension): ")
        create_qr(data, filename)

    # Option 2: Decode QR code from a saved image
    elif choice == "2":
        filename = input("Enter QR image filename to decode: ")
        decode_qr_image(filename)

    # Option 3: Decode QR code using live webcam
    elif choice == "3":
        decode_qr_camera()

    # Option 4: Exit the program
    else:
        print("Exiting...")
