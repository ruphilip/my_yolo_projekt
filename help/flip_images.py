"""
Sucht im angegebenen Ordner alle .jpg- und .jpeg-Bilder, die noch kein "_sp"
im Namen haben, dreht sie um 180 Grad und speichert sie als
<name>_sp.<endung>.

Die Originale bleiben unverändert.

Voraussetzung:  pip install pillow
"""

from pathlib import Path

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# HIER DEN PFAD ZUM ORDNER MIT DEN BILDERN EINTRAGEN
#
# Windows-Beispiel:  r"C:\Users\Name\Bilder\Muenzen"
# Mac/Linux-Beispiel: "/Users/name/Bilder/Muenzen"
#
# Leer lassen ("") = der Ordner, in dem dieses Skript liegt.
# ---------------------------------------------------------------------------
ORDNER_PFAD = r"/Users/philipprutimann/Documents/ZHAW/CAS_ADML/MLOps/Projekt/MBilder"

EXTENSIONS = {".jpg", ".jpeg"}
SUFFIX = "_sp"


def main() -> None:
    if ORDNER_PFAD.strip():
        folder = Path(ORDNER_PFAD.strip())
    else:
        folder = Path(__file__).resolve().parent

    if not folder.is_dir():
        print(f"Ordner nicht gefunden: {folder}")
        return

    print(f"Suche in: {folder}\n")

    count = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue
        if SUFFIX in path.stem:
            continue

        target = path.with_name(f"{path.stem}{SUFFIX}{path.suffix}")

        with Image.open(path) as img:
            # Handy-Fotos haben oft eine EXIF-Drehung; erst anwenden,
            # damit um 180 Grad so gedreht wird, wie das Bild angezeigt wird.
            img = ImageOps.exif_transpose(img)
            rotated = img.transpose(Image.Transpose.ROTATE_180)
            rotated.save(target, quality=95)

        print(f"{path.name}  ->  {target.name}")
        count += 1

    print(f"\nFertig: {count} Bild(er) um 180 Grad gedreht.")


if __name__ == "__main__":
    main()