#!/bin/bash

# 1. Benutzer nach einer Beschreibung für das Update fragen
echo "Bitte gib eine kurze Beschreibung für diese Version ein:"
read -r commit_msg

if [ -z "$commit_msg" ]; then
    echo "Fehler: Beschreibung darf nicht leer sein! Abbruch."
    exit 1
fi

# 2. Letztes Git-Tag holen, um die Major-Versionsnummer automatisch zu erhöhen
# Wenn noch kein Tag existiert, starten wir bei v1.0
last_tag=$(git describe --tags --abbrev=0 2>/dev/null)

if [ -z "$last_tag" ]; then
    next_tag="v1.0"
else
    # Extrahiert die Major-Zahl vor dem Punkt und erhöht sie um 1
    major=$(echo "$last_tag" | sed 's/v//' | cut -d. -f1)
    new_major=$((major + 1))
    next_tag="v${new_major}.0"
fi

echo "🚀 Starte Daten-Versioning für Major-Version: $next_tag..."

# 3. DVC: Änderungen scannen und lokalen Cache aktualisieren
dvc add dataset/

# 4. DVC: Daten physisch in dein lokales Archiv pushen
dvc push

# 5. Git: Geänderte Pointer-Datei commiten
git commit -am "Dataset $next_tag: $commit_msg"

# 6. Git: Neues Versionstag setzen
git tag -a "$next_tag" -m "$commit_msg"

echo "Fertig! Datensatz wurde unter '$next_tag' lokal archiviert und in Git getaggt."
