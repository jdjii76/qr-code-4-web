import io
import streamlit as st
import qrcode
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="QR Tool", page_icon="🔍")

st.title("Accessible QR Code Generator & Scanner")

# --- GENERATOR SECTION ---
st.subheader("1. Generate QR Code")
col1, col2 = st.columns([2, 1])

with col1:
    data = st.text_input("QR Code Content (text or URL)", placeholder="https://example.com")
    fill = st.color_picker("Choose QR Color", "#000000")

if data.strip():
    # Improved QR generation with better sizing
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data.strip())
    qr.make(fit=True)
    
    # Create the image
    img = qr.make_image(fill_color=fill, back_color="white").convert("RGB")
    
    # Display the result
    st.image(img, caption="Right-click to copy or use the button below", width=300)

    # Prepare download buffer
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.download_button(
        label="📥 Download QR Code (PNG)",
        data=buf.getvalue(),
        file_name="qrcode.png",
        mime="image/png"
    )

st.divider()

# --- SCANNER SECTION ---
st.subheader("2. Scan QR Code from Image")

uploaded = st.file_uploader("Upload a QR code image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", width=250)

        # Convert RGB (PIL) to BGR (OpenCV)
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Use WeChatQRCodeDetector or standard QRCodeDetector
        # WeChat detector is often more robust, but standard is fine for basic use
        detector = cv2.QRCodeDetector()
        val, points, straight_qrcode = detector.detectAndDecode(img_bgr)

        if val:
            st.success("**Decoded Content:**")
            st.code(val, language=None)
            if val.startswith("http"):
                st.info(f"🔗 [Click here to open link]({val})")
        else:
            st.warning("Could not find a valid QR code. Try a clearer image.")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")