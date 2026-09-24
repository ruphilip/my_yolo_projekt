import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Titel und Beschreibung der Web-App
st.set_page_config(page_title="YOLOv8 Live-Webcam", layout="centered")
st.title("📹 YOLOv8 Live-Webcam Erkennung")
st.write("Nutze die Webcam deines Macs für eine Objekterkennung in Echtzeit.")

# Seitenleiste für Einstellungen
st.sidebar.header("🔧 Einstellungen")
conf_threshold = st.sidebar.slider("Sicherheitsschwelle (Confidence)", 0.0, 1.0, 0.25, 0.05)

# 2. Pfad zum trainierten Modell
MODEL_PATH = "./mlruns/1/d38a5f6388444f16b614b9ab0f597c45/artifacts/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    st.sidebar.success("Modell geladen!")
except Exception as e:
    st.sidebar.error(f"Modell-Fehler: {e}")
    st.stop()

# 3. Start/Stopp-Button für die Webcam
run_webcam = st.checkbox("▶️ Webcam starten / stoppen")

# Platzhalter für den Live-Videostream im Browser
frame_placeholder = st.empty()

# 4. Webcam-Schleife
if run_webcam:
    # Index 0 ist standardmässig die integrierte FaceTime-Kamera des Macs
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Konnte nicht auf die Webcam zugreifen. Bitte überprüfe die Berechtigungen.")
        run_webcam = False

    while cap.isOpened() and run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.write("Video-Feed abgebrochen.")
            break

        # YOLO-Vorhersage auf dem aktuellen Frame ausführen (mit Mac MPS-Beschleunigung)
        results = model.predict(source=frame, device="mps", conf=conf_threshold, verbose=False)
        
        # Bounding Boxes direkt auf das Frame zeichnen
        res_plotted = results[0].plot()
        
        # OpenCV liest in BGR, Streamlit benötigt RGB
        frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Das Array in ein Pillow-Bild umwandeln
        img_stream = Image.fromarray(frame_rgb)
        
        # Den Platzhalter im Browser mit dem aktuellen Frame füllen
        frame_placeholder.image(img_stream, use_container_width=True)

    # Kamera wieder freigeben, wenn der Button deaktiviert wird
    cap.release()
    frame_placeholder.empty()
    st.info("Webcam wurde gestoppt.")
else:
    st.info("Setze das Häkchen oben, um den Live-Videostream zu starten.")
