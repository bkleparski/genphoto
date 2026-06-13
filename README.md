# GenPhoto — AI Photo Generation Studio

Samodzielny serwer do generowania **fotorealistycznych zdjęć** przy użyciu Stable Diffusion Forge i DeepSeek AI.  
Napisany w czystym Pythonie — bez frameworków, bez zewnętrznych zależności.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Funkcje

- **Presety** — gotowe konfiguracje pod Portrait, Fashion, Landscape, Architecture i Street Photography
- **AI Prompt** — opisujesz scenę po polsku, DeepSeek generuje optymalny prompt pozytywny i negatywny
- **Tryb zaawansowany** — pełna kontrola: model, sampler, scheduler, steps, CFG, seed, wymiary, batch
- **Historia generacji** — SQLite, podgląd miniatur, ponowne użycie parametrów, usuwanie wpisów
- **Lightbox** — pełnoekranowy podgląd wygenerowanych zdjęć z nawigacją klawiaturą
- **Logowanie** — sesja cookie, hasło przechowywane jako SHA-256
- **Integracja z Forge REST API** — `POST /sdapi/v1/txt2img`, polling statusu generowania
- **Bez zależności** — wyłącznie biblioteka standardowa Pythona 3.9+

---

## Wymagania

- Python 3.9+
- [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) uruchomiony lokalnie (`--api`)
- Konto [DeepSeek](https://platform.deepseek.com/) — klucz API do generowania promptów

---

## Instalacja

### 1. Pobierz skrypt

```bash
wget https://raw.githubusercontent.com/bkleparski/genphoto/main/genphoto.py
```

### 2. Skonfiguruj zmienne środowiskowe

```bash
cp .env.example ~/.env.genphoto
```

Edytuj `~/.env.genphoto`:

```env
GP_USERNAME=admin
GP_PASSWORD_HASH=<hash SHA-256 twojego hasła>
GP_OUTPUTS_DIR=/home/user/stable-diffusion-webui/outputs
GP_FORGE_URL=http://localhost:7860
GP_PORT=7862
GP_DB_PATH=/home/user/genphoto.db
GP_DEEPSEEK_KEY=<klucz z platform.deepseek.com>
GP_DEEPSEEK_MODEL=deepseek-v4-flash
```

Wygeneruj hash hasła:

```bash
python3 -c "import hashlib; print(hashlib.sha256(b'twoje-haslo').hexdigest())"
```

### 3. Uruchom Forge z API

```bash
./run.sh --api
```

### 4. Uruchom GenPhoto

```bash
source ~/.env.genphoto && python3 genphoto.py
```

Studio dostępne pod `http://localhost:7862`

---

## Uruchomienie jako usługa systemd (Linux)

```bash
cp genphoto.service.example ~/.config/systemd/user/genphoto.service
# uzupełnij ~/.env.genphoto

systemctl --user daemon-reload
systemctl --user enable genphoto.service
systemctl --user start genphoto.service
```

---

## Zmienne środowiskowe

| Zmienna | Domyślna | Opis |
|---|---|---|
| `GP_USERNAME` | `admin` | Login użytkownika |
| `GP_PASSWORD_HASH` | (wymagane) | SHA-256 hasła |
| `GP_OUTPUTS_DIR` | `/home/user/stable-diffusion-webui/outputs` | Katalog outputów Forge |
| `GP_FORGE_URL` | `http://localhost:7860` | Adres Forge API |
| `GP_PORT` | `7862` | Port serwera GenPhoto |
| `GP_DB_PATH` | `/home/user/genphoto.db` | Plik bazy danych historii |
| `GP_DEEPSEEK_KEY` | (wymagane do AI Prompt) | Klucz API DeepSeek |
| `GP_DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model DeepSeek |
| `GP_GALLERY_URL` | `` | Opcjonalny link do galerii plików |

---

## Presety

| Preset | Model | Wymiary | Zastosowanie |
|---|---|---|---|
| Portrait | RealVisXL_V5_fp16 | 832×1216 | Portrety, twarze |
| Fashion | RealVisXL_V5_fp16 | 832×1216 | Moda, postaci |
| Landscape | RealVisXL_V5_fp16 | 1216×832 | Krajobrazy, plenery |
| Architecture | RealVisXL_V5_fp16 | 1024×1024 | Budynki, wnętrza |
| Street | RealVisXL_V5_fp16 | 896×1152 | Fotografia uliczna |

Modele muszą być zainstalowane w Forge. Nazwy checkpointów można zmienić bezpośrednio w kodzie w sekcji `PRESETS`.

---

## Bezpieczeństwo

- Hasło **nigdy** nie jest przechowywane wprost — tylko hash SHA-256
- Plik `.env` z hasłem i kluczem API jest wykluczony z git przez `.gitignore`
- Sesja oparta o losowy token `secrets.token_hex(32)` z cookie `HttpOnly; SameSite=Lax`
- Wszystkie ścieżki plików są weryfikowane relative do `OUTPUTS_DIR`

---

## Licencja

MIT — możesz używać, modyfikować i dystrybuować dowolnie.
