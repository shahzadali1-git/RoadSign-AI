
from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="RoadSign AI",
    page_icon="🚦",
    layout="wide"
)

MODEL_PATH = Path(
    "runs/detect/runs/roadvision/yolo11_30epochs/weights/best.pt"
)

# ---------------- DESIGN ----------------

st.title("🚦 RoadSign AI")
st.subheader("Intelligent Traffic Sign Detection System")

st.write(
    "Upload a road image to detect traffic signs using "
    "a custom-trained YOLO11n computer vision model."
)

st.divider()

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

model = load_model()

# ---------------- SIDEBAR ----------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

st.sidebar.success("YOLO11n model loaded")

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # ---------------- DETECTION ----------------

    with st.spinner("Detecting traffic signs..."):

        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False
        )

    result = results[0]

    annotated_image = result.plot()
    annotated_image = annotated_image[:, :, ::-1]

    boxes = result.boxes

    with col2:
        st.subheader("Detection Results")
        st.image(
            annotated_image,
            channels="RGB",
            use_container_width=True
        )

    # ---------------- METRICS ----------------

    st.divider()
    st.subheader("Detection Summary")

    total_detections = len(boxes)

    st.metric("Total Signs Detected", total_detections)

    if total_detections > 0:

        detection_data = []

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            score = float(box.conf[0])

            detection_data.append({
                "Traffic Sign": class_name,
                "Confidence": f"{score * 100:.2f}%"
            })

        df = pd.DataFrame(detection_data)

        st.dataframe(df, use_container_width=True)

    else:
        st.warning("No traffic signs detected. Try lowering the confidence threshold.")

    # ---------------- DOWNLOAD ----------------

        import io

    image_buffer = io.BytesIO()
    Image.fromarray(annotated_image).save(image_buffer, format="PNG")

    st.download_button(
        label="Download Detection Image",
        data=image_buffer.getvalue(),
        file_name="roadsign_detection.png",
        mime="image/png"
    )

else:
    st.info("Upload an image to start traffic sign detection.")
