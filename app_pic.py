import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Titel und Beschreibung der Web-App
st.set_page_config(page_title="YOLOv8 Objektkennung", layout="centered")
st.title("🔍 YOLOv8 Objektkennung")
st.write("Lade ein Bild hoch, um Objekte mit deinem lokal trainierten Modell zu erkennen.")

# 2. Pfad zum trainierten Modell (Passe den Pfad an dein 'best.pt' an)
MODEL_PATH = "./mlruns/1/d38a5f6388444f16b614b9ab0f597c45/artifacts/weights/best.pt"


# Modell laden (wird gecacht, damit es nicht bei jedem Klick neu geladen wird)
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    st.success("Modell erfolgreich geladen!")
except Exception as e:
    st.error(f"Fehler beim Laden des Modells. Prüfe den Pfad: {e}")
    st.stop()

# 3. Datei-Uploader für den Benutzer
uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild mit Pillow öffnen
    image = Image.open(uploaded_file)
    
    # Layout aufteilen: Links Original, rechts Vorhersage
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Originalbild")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Objektkennung")
        
        # Spinner anzeigen, während das Modell rechnet
        with st.spinner("Modell analysiert das Bild..."):
            # Vorhersage auf der Mac-GPU (mps) ausführen
            results = model.predict(source=image, device="mps")
            
            # Das erste Ergebnis holen (da wir nur ein Bild senden)
            res = results[0]
            
            # Das Ergebnis-Bild mit den gezeichneten Bounding Boxes als NumPy-Array holen
            res_plotted = res.plot()
            
            # Von BGR (OpenCV) zu RGB (Streamlit/Pillow) konvertieren
            res_image = Image.fromarray(res_plotted[..., ::-1])
            
            # Bild in der App anzeigen
            st.image(res_image, use_container_width=True)
            
    # 4. Gefundene Objekte als Text auflisten
    st.subheader("Erkannte Objekte Details:")
    boxes = res.boxes
    if len(boxes) == 0:
        st.write("Keine Objekte erkannt.")
    else:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0]) * 100
            st.write(f"🎯 **{class_name}** mit {confidence:.1f}% Sicherheit")
