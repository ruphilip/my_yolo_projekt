import sys
from pathlib import Path

ENDUNGEN = {".jpg", ".jpeg"}

# True = nur anzeigen, was umbenannt würde. Auf False setzen, um wirklich umzubenennen.
DRY_RUN = False


def main(ordner: str | None = None) -> None:
    # Ordner als Argument übergeben, sonst aktueller Ordner
    if ordner is not None:
        ordner = Path(ordner)
    else:
        ordner = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if not ordner.is_dir():
        print(f"Fehler: '{ordner}' ist kein gültiger Ordner.")
        sys.exit(1)

    bilder = sorted(
        p for p in ordner.iterdir()
        if p.is_file() and p.suffix.lower() in ENDUNGEN
    )

    # Dateien mit reiner Zahl als Namen -> bestimmen die höchste Nummer
    max_bild = max((int(p.stem) for p in bilder if p.stem.isdigit()), default=0)

    # Alle anderen werden neu nummeriert
    umzubenennen = [p for p in bilder if not p.stem.isdigit()]

    naechste = max_bild + 1
    for pfad in umzubenennen:
        ziel = pfad.with_name(f"{naechste}{pfad.suffix}")
        print(f"{pfad.name} -> {ziel.name}")
        if not DRY_RUN:
            if ziel.exists():
                print(f"  Übersprungen: '{ziel.name}' existiert bereits.")
                continue
            pfad.rename(ziel)
        naechste += 1

    if DRY_RUN:
        print("\nTrockenlauf: nichts wurde geändert (DRY_RUN = False setzen).")


if __name__ == "__main__":
    main("/Users/philipprutimann/Documents/ZHAW/CAS_ADML/MLOps/Projekt/MBilder/")