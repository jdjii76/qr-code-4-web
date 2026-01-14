import io
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st
import qrcode
from qrcode.image.svg import SvgImage
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="QR Tool Pro", page_icon="🔍", layout="wide")
st.title("QR Tool Pro — Generator, Templates, Batch, and Scanner")

# ----------------------------
# Helpers
# ----------------------------
ERROR_LEVELS = {
    "L (7%)": qrcode.constants.ERROR_CORRECT_L,
    "M (15%)": qrcode.constants.ERROR_CORRECT_M,
    "Q (25%)": qrcode.constants.ERROR_CORRECT_Q,
    "H (30%)": qrcode.constants.ERROR_CORRECT_H,
}

@dataclass
class QRSettings:
    fill_color: str
    back_color: str
    box_size: int
    border: int
    error_correction: int

def build_wifi_payload(ssid: str, password: str, security: str, hidden: bool) -> str:
    # WIFI:T:WPA;S:myssid;P:mypass;H:true;;
    sec = security if security != "None" else "nopass"
    h = "true" if hidden else "false"
    return f"WIFI:T:{sec};S:{ssid};P:{password};H:{h};;"

def build_mailto_payload(email: str, subject: str, body: str) -> str:
    # Basic mailto; URL-encoding can be added if needed
    pieces = [f"mailto:{email}"]
    params = []
    if subject.strip():
        params.append(f"subject={subject.strip()}")
    if body.strip():
        params.append(f"body={body.strip()}")
    if params:
        pieces.append("?" + "&".join(params))
    return "".join(pieces)

def build_sms_payload(phone: str, message: str) -> str:
    # Common format
    if message.strip():
        return f"SMSTO:{phone}:{message}"
    return f"SMSTO:{phone}:"

def build_geo_payload(lat: float, lon: float) -> str:
    return f"geo:{lat},{lon}"

def build_vcard_payload(
    first: str,
    last: str,
    org: str,
    title: str,
    phone: str,
    email: str,
    url: str
) -> str:
    # vCard 3.0 simple
    lines = [
        "BEGIN:VCARD",
        "VERSION:3.0",
        f"N:{last};{first};;;",
        f"FN:{first} {last}".strip(),
    ]
    if org.strip():
        lines.append(f"ORG:{org}")
    if title.strip():
        lines.append(f"TITLE:{title}")
    if phone.strip():
        lines.append(f"TEL;TYPE=CELL:{phone}")
    if email.strip():
        lines.append(f"EMAIL:{email}")
    if url.strip():
        lines.append(f"URL:{url}")
    lines.append("END:VCARD")
    return "\n".join(lines)

def make_qr_image(data: str, settings: QRSettings) -> Image.Image:
    qr = qrcode.QRCode(
        version=None,  # auto
        error_correction=settings.error_correction,
        box_size=settings.box_size,
        border=settings.border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color=settings.fill_color, back_color=settings.back_color).convert("RGBA")
    return img

def add_logo(qr_img_rgba: Image.Image, logo_img: Image.Image, logo_scale_pct: int) -> Image.Image:
    """
    Places a logo at center. Keep scale modest (e.g., <= 25%) and use high error correction for best results.
    """
    qr = qr_img_rgba.copy()
    W, H = qr.size

    logo = logo_img.convert("RGBA")

    # scale is percent of QR width
    target_w = int(W * (logo_scale_pct / 100.0))
    if target_w <= 0:
        return qr

    # preserve aspect ratio
    ratio = target_w / logo.size[0]
    target_h = int(logo.size[1] * ratio)
    logo = logo.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # center paste
    x = (W - target_w) // 2
    y = (H - target_h) // 2
    qr.alpha_composite(logo, (x, y))
    return qr

def img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def channel_lum(channel: int) -> float:
        srgb = channel / 255.0
        if srgb <= 0.03928:
            return srgb / 12.92
        return ((srgb + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel_lum(r) + 0.7152 * channel_lum(g) + 0.0722 * channel_lum(b)

def contrast_ratio(color_a: str, color_b: str) -> float:
    lum_a = relative_luminance(hex_to_rgb(color_a))
    lum_b = relative_luminance(hex_to_rgb(color_b))
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)

def render_contrast_guidance(foreground: str, background: str, label: str) -> None:
    ratio = contrast_ratio(foreground, background)
    st.caption(f"{label} contrast ratio: **{ratio:.2f}:1**")
    if ratio < 4.5:
        st.warning(
            "Low contrast can make QR codes harder to scan. Aim for at least **4.5:1** contrast.",
            icon="⚠️",
        )

def make_qr_svg_bytes(data: str, settings: QRSettings) -> bytes:
    qr = qrcode.QRCode(
        version=None,
        error_correction=settings.error_correction,
        box_size=settings.box_size,
        border=settings.border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    svg_img = qr.make_image(image_factory=SvgImage, fill_color=settings.fill_color, back_color=settings.back_color)
    # svg_img is an ElementTree-like; save to bytes
    out = io.BytesIO()
    svg_img.save(out)
    return out.getvalue()

def detect_multiple_qr(img_bgr: np.ndarray) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Returns list of (decoded_text, points) where points are 4 corners.
    Uses OpenCV QRCodeDetector multi method if available.
    """
    detector = cv2.QRCodeDetector()

    # Newer OpenCV provides detectAndDecodeMulti
    if hasattr(detector, "detectAndDecodeMulti"):
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img_bgr)
        results = []
        if ok and decoded_info is not None:
            for txt, pts in zip(decoded_info, points if points is not None else []):
                if txt:
                    results.append((txt, pts))
        return results

    # Fallback: single
    val, points, _ = detector.detectAndDecode(img_bgr)
    if val:
        return [(val, points)]
    return []

def draw_boxes(img_rgb: np.ndarray, detections: List[Tuple[str, Optional[np.ndarray]]]) -> np.ndarray:
    out = img_rgb.copy()
    for _, pts in detections:
        if pts is None:
            continue
        pts = pts.astype(int).reshape(-1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    return out

# ----------------------------
# Layout
# ----------------------------
tab_gen, tab_batch, tab_scan = st.tabs(["Generator", "Batch", "Scanner"])

# ----------------------------
# Generator Tab
# ----------------------------
with tab_gen:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("1) Choose a QR type")
        qr_type = st.selectbox(
            "Template",
            ["Text / URL", "Wi-Fi", "vCard", "Email", "SMS", "Geo (lat/lon)"],
            index=0,
            help="Pick the kind of information you want to encode."
        )

        payload = ""

        if qr_type == "Text / URL":
            payload = st.text_area(
                "QR Content",
                placeholder="https://example.com",
                height=120,
                help="Paste a URL or any text you want to encode.",
            ).strip()

        elif qr_type == "Wi-Fi":
            ssid = st.text_input("SSID", help="The Wi-Fi network name.")
            security = st.selectbox(
                "Security",
                ["WPA", "WEP", "None"],
                index=0,
                help="Choose the Wi-Fi security type.",
            )
            password = st.text_input(
                "Password",
                type="password" if security != "None" else "default",
                help="Required for WPA/WEP networks.",
            )
            hidden = st.checkbox("Hidden network", value=False, help="Check if the SSID is hidden.")
            if ssid.strip():
                payload = build_wifi_payload(ssid.strip(), password, security, hidden)

        elif qr_type == "vCard":
            c1, c2 = st.columns(2)
            with c1:
                first = st.text_input("First name", help="Given name.")
                last = st.text_input("Last name", help="Family name.")
                org = st.text_input("Organization", help="Company or organization name.")
            with c2:
                title = st.text_input("Title", help="Role or job title.")
                phone = st.text_input("Phone", help="Mobile or main contact number.")
                email = st.text_input("Email", help="Email address.")
            url = st.text_input("Website (optional)", help="Include a website URL if needed.")
            if (first.strip() or last.strip() or phone.strip() or email.strip()):
                payload = build_vcard_payload(first, last, org, title, phone, email, url)

        elif qr_type == "Email":
            email = st.text_input("To", help="Recipient email address.")
            subject = st.text_input("Subject", help="Email subject line.")
            body = st.text_area("Body", height=100, help="Email body content.")
            if email.strip():
                payload = build_mailto_payload(email.strip(), subject, body)

        elif qr_type == "SMS":
            phone = st.text_input("Phone number", help="Include country code if needed.")
            message = st.text_area("Message", height=100, help="SMS message content.")
            if phone.strip():
                payload = build_sms_payload(phone.strip(), message)

        elif qr_type == "Geo (lat/lon)":
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input(
                    "Latitude",
                    value=34.000000,
                    format="%.6f",
                    help="Use decimal degrees, e.g., 34.000000",
                )
            with c2:
                lon = st.number_input(
                    "Longitude",
                    value=-84.000000,
                    format="%.6f",
                    help="Use decimal degrees, e.g., -84.000000",
                )
            payload = build_geo_payload(lat, lon)

        st.caption("Tip: For logos and dense data (vCard), use error correction **H** and a larger box size.")

    with right:
        st.subheader("2) Style & output")
        fill = st.color_picker("QR color", "#000000", help="Choose a dark foreground color.")
        back = st.color_picker("Background color", "#FFFFFF", help="Choose a light background color.")
        render_contrast_guidance(fill, back, "QR")
        error_label = st.selectbox(
            "Error correction",
            list(ERROR_LEVELS.keys()),
            index=3,
            help="Higher levels improve scan reliability but reduce capacity.",
        )
        box_size = st.slider(
            "Box size",
            4,
            20,
            10,
            help="Controls the size of each QR module (pixel).",
        )
        border = st.slider(
            "Border (quiet zone)",
            1,
            10,
            4,
            help="Keep at least 4 modules for reliable scanning.",
        )

        add_logo_toggle = st.checkbox(
            "Add center logo",
            value=False,
            help="Logos can reduce scan reliability, use higher error correction.",
        )
        logo_scale = st.slider("Logo size (% of QR width)", 10, 35, 22, disabled=not add_logo_toggle)
        logo_file = st.file_uploader(
            "Logo image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            disabled=not add_logo_toggle,
            help="Transparent PNGs work best.",
        )

        output_format = st.radio(
            "Download format",
            ["PNG", "SVG"],
            horizontal=True,
            help="SVG is vector-based and scales without blurring.",
        )

        settings = QRSettings(
            fill_color=fill,
            back_color=back,
            box_size=box_size,
            border=border,
            error_correction=ERROR_LEVELS[error_label],
        )

        if payload:
            try:
                # Generate base image (PNG preview uses raster)
                img = make_qr_image(payload, settings)

                if add_logo_toggle and logo_file is not None:
                    logo_img = Image.open(logo_file)
                    img = add_logo(img, logo_img, logo_scale)

                st.image(img.convert("RGB"), caption="Preview", width=320)

                if output_format == "PNG":
                    png_bytes = img_to_png_bytes(img.convert("RGBA"))
                    st.download_button(
                        "📥 Download PNG",
                        data=png_bytes,
                        file_name="qrcode.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    # SVG generation (logo overlay not included; SVG is vector)
                    svg_bytes = make_qr_svg_bytes(payload, settings)
                    st.download_button(
                        "📥 Download SVG",
                        data=svg_bytes,
                        file_name="qrcode.svg",
                        mime="image/svg+xml",
                        use_container_width=True,
                    )

                with st.expander("Show encoded payload"):
                    st.code(payload, language=None)

            except Exception as e:
                st.error(f"QR generation error: {e}")
        else:
            st.info("Enter content above to generate a QR code.")

# ----------------------------
# Batch Tab
# ----------------------------
with tab_batch:
    st.subheader("Batch QR generation (CSV → ZIP)")
    st.write("Upload a CSV with at least one column named `data`. Optional: `filename` column.")

    sample = pd.DataFrame(
        {"data": ["https://example.com/a", "https://example.com/b"],
         "filename": ["station_a", "station_b"]}
    )
    with st.expander("CSV format example"):
        st.dataframe(sample, use_container_width=True)

    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    colA, colB, colC = st.columns(3)
    with colA:
        b_fill = st.color_picker("QR color", "#000000", key="b_fill", help="Choose a dark foreground color.")
    with colB:
        b_back = st.color_picker("Background", "#FFFFFF", key="b_back", help="Choose a light background color.")
    with colC:
        b_error_label = st.selectbox(
            "Error correction",
            list(ERROR_LEVELS.keys()),
            index=3,
            key="b_err",
            help="Higher levels improve scan reliability but reduce capacity.",
        )

    render_contrast_guidance(b_fill, b_back, "Batch QR")
    b_box = st.slider("Box size", 4, 20, 10, key="b_box", help="Controls module size.")
    b_border = st.slider("Border", 1, 10, 4, key="b_border", help="Keep at least 4 modules.")

    batch_settings = QRSettings(
        fill_color=b_fill,
        back_color=b_back,
        box_size=b_box,
        border=b_border,
        error_correction=ERROR_LEVELS[b_error_label],
    )

    if batch_file is not None:
        try:
            df = pd.read_csv(batch_file)
            if "data" not in df.columns:
                st.error("CSV must include a `data` column.")
            else:
                fmt = st.radio(
                    "Output",
                    ["PNG", "SVG"],
                    horizontal=True,
                    key="b_fmt",
                    help="SVG is vector-based and scales without blurring.",
                )
                make_zip = st.button("Build ZIP", type="primary")

                if make_zip:
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                        for i, row in df.iterrows():
                            data = str(row["data"]).strip()
                            if not data:
                                continue
                            name = None
                            if "filename" in df.columns:
                                name = str(row.get("filename", "")).strip() or None
                            if not name:
                                name = f"qr_{i+1}"

                            if fmt == "PNG":
                                img = make_qr_image(data, batch_settings).convert("RGBA")
                                z.writestr(f"{name}.png", img_to_png_bytes(img))
                            else:
                                z.writestr(f"{name}.svg", make_qr_svg_bytes(data, batch_settings))

                    st.download_button(
                        "📦 Download ZIP",
                        data=zbuf.getvalue(),
                        file_name="qr_batch.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Batch error: {e}")
    else:
        st.info("Upload a CSV to enable batch generation.")

# ----------------------------
# Scanner Tab
# ----------------------------
with tab_scan:
    st.subheader("Scan QR code(s) from an image")
    uploaded = st.file_uploader("Upload a PNG/JPG", type=["png", "jpg", "jpeg"], key="scan_img")

    if uploaded is not None:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            detections = detect_multiple_qr(img_bgr)
            boxed = draw_boxes(img_rgb, detections) if detections else img_rgb

            st.image(boxed, caption="Detected QR codes (boxed when found)", use_container_width=True)

            if detections:
                st.success(f"Found {len(detections)} QR code(s).")
                for idx, (val, _) in enumerate(detections, start=1):
                    st.markdown(f"**#{idx} Decoded content**")
                    st.code(val, language=None)
                    if val.startswith("http://") or val.startswith("https://"):
                        st.link_button("Open link", val)
                    st.divider()
            else:
                st.warning("No QR codes found. Try a clearer image, higher resolution, or less glare.")

        except Exception as e:
            st.error(f"Scan error: {e}")
