````markdown
# DVC + Google Drive Remote (GCP VM) — Full Setup & Troubleshooting

This document explains how to configure DVC to use a Google Drive folder as a remote on a Google Cloud VM.
It is written to be reproducible when changing machines/environments (new VM, new laptop, new repo clone).

> Context: This repo uses DVC with a Google Drive remote folder ID:
> `gdrive://1YsCkh_UE62q3IAI96Uo_u94iVAEF4G2S`

---

## 0) Prerequisites

- A Google Drive folder created for DVC storage (you need its folder ID).
- A Google Cloud project (recommended) with **Google Drive API enabled**.
- DVC installed with GDrive support:
  - `pip install "dvc[gdrive]"`

---

## 1) Enable Google Drive API (GCP Project)

From a machine authenticated to GCP (Cloud Shell or local with permissions):

```bash
gcloud config set project <PROJECT_ID>
gcloud services enable drive.googleapis.com
gcloud services list --enabled | grep -i drive
````

Expected:

* `drive.googleapis.com` appears in enabled services.

---

## 2) Create OAuth client credentials (Desktop App)

In Google Cloud Console (same project):

1. APIs & Services → Credentials → Create Credentials → OAuth client ID
2. Application type: **Desktop app**
3. Save:

   * `client_id`
   * `client_secret`

Security note:

* These are the **OAuth app credentials** (not the user token). Do not commit them to Git.
* If the secret was exposed, rotate it in the Cloud Console and update DVC config accordingly.

---

## 3) DVC remote configuration in the repo (versioned)

In the repository root on the VM:

```bash
cd ~/clusterllm-local
dvc init   # only if not already initialized
```

Add the Google Drive remote and set it as default:

```bash
dvc remote add -d myremote gdrive://1YsCkh_UE62q3IAI96Uo_u94iVAEF4G2S
```

Set OAuth client credentials in `.dvc/config` (project-level config):

```bash
dvc remote modify myremote gdrive_client_id "<CLIENT_ID>"
dvc remote modify myremote gdrive_client_secret "<CLIENT_SECRET>"
```

Verify:

```bash
dvc remote list
cat .dvc/config
```

Expected `.dvc/config` snippet:

```ini
[core]
    remote = myremote

[remote "myremote"]
    url = gdrive://1YsCkh_UE62q3IAI96Uo_u94iVAEF4G2S
    gdrive_client_id = ...
    gdrive_client_secret = ...
```

> IMPORTANT: DVC config files must use INI headers like `[remote "myremote"]`.
> Avoid broken headers like `['remote "myremote"']` (these caused hard-to-debug errors).

---

## 4) Store user token locally (NOT committed)

DVC stores user tokens locally after OAuth login.
We force DVC to write credentials to a safe path in the home directory:

```bash
mkdir -p ~/.gdrive
dvc remote modify myremote --local gdrive_user_credentials_file ~/.gdrive/myremote-credentials.json
```

Ignore local credentials in Git:

```bash
echo ".gdrive/" >> .gitignore
echo ".dvc/tmp/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore local gdrive credentials" || true
```

---

## 5) Why OAuth was difficult on the VM (TTY / stdin issue)

On some GCP VM terminals (especially code-server / non-interactive shells), `stdin` may not be a TTY:

```bash
python - <<'PY'
import sys
print("stdin isatty:", sys.stdin.isatty())
print("stdout isatty:", sys.stdout.isatty())
PY
```

If `stdin isatty: False`, interactive OAuth flows can fail or loop indefinitely.
In that case, the most reliable approach is:

✅ Generate `creds.json` on a local computer (Windows) where the browser auth works
✅ Upload it to the VM and use it as `~/.gdrive/myremote-credentials.json`

---

## 6) Generate `creds.json` on Windows (robust method)

### 6.1 Create `oauth.json` without UTF-8 BOM (PowerShell)

BOM caused JSON parsing failures with oauth2client. Use this exact method:

```powershell
cd C:\Users\MLSD

$json = '{"installed":{"client_id":"<CLIENT_ID>","project_id":"<PROJECT_ID>","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"<CLIENT_SECRET>","redirect_uris":["http://localhost"]}}'

[System.IO.File]::WriteAllText("$PWD\oauth.json", $json, (New-Object System.Text.UTF8Encoding($false)))
python -c "import json; json.load(open('oauth.json','r',encoding='utf-8')); print('oauth.json OK')"
```

### 6.2 Install pydrive2

```powershell
python -m pip install -U pydrive2
```

### 6.3 Create `make_creds.py`

```powershell
notepad .\make_creds.py
```

Paste:

```python
from pydrive2.auth import GoogleAuth

gauth = GoogleAuth(settings={
    "client_config_backend": "file",
    "client_config_file": "oauth.json",
    "save_credentials": True,
    "save_credentials_backend": "file",
    "save_credentials_file": "creds.json",
    "get_refresh_token": True,
    "oauth_scope": [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.appdata",
    ],
})
gauth.LocalWebserverAuth()
print("Saved creds.json")
```

Run:

```powershell
Remove-Item -Force .\creds.json -ErrorAction SilentlyContinue
python .\make_creds.py
dir .\creds.json
```

Expected:

* `creds.json` created (≈ 1–3 KB)

---

## 7) Copy `creds.json` to the VM

### 7.1 Cloud Shell method (recommended when Windows gcloud has IAM issues)

1. Open Cloud Shell (project: `<PROJECT_ID>`)
2. Ensure gcloud is authenticated:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud auth list
```

3. Upload `creds.json` to Cloud Shell (UI: Upload file)
4. Confirm:

```bash
ls -lh ~/creds.json
```

5. Copy to VM (replace zone):

```bash
gcloud compute scp ~/creds.json <VM_USER>@<VM_NAME>:/home/<VM_USER>/.gdrive/myremote-credentials.json \
  --zone <ZONE>
```

Example:

* VM name: `hamady-gpu-l4`
* zone: `europe-west4-c`
* user: `hamadygackou777`

```bash
gcloud compute scp ~/creds.json hamadygackou777@hamady-gpu-l4:/home/hamadygackou777/.gdrive/myremote-credentials.json \
  --zone europe-west4-c
```

### 7.2 VM permissions + verify

On the VM:

```bash
chmod 700 ~/.gdrive
chmod 600 ~/.gdrive/myremote-credentials.json
ls -lh ~/.gdrive/myremote-credentials.json
```

---

## 8) Final test: DVC push/pull

```bash
cd ~/clusterllm-local
dvc push -v
```

Expected:

* No browser URL
* Successful push (example output: `N files pushed`)

Test pull:

```bash
dvc pull -v
```

---

## 9) Troubleshooting

### 9.1 “expected 'url' for dictionary value @ data['remote']['gdrive']”

Cause:

* A broken remote section exists in one of:

  * `.dvc/config.local`
  * `~/.config/dvc/config`
  * `/etc/xdg/dvc/config`

Fix:

* Search and remove invalid remotes, especially `[remote "gdrive"]` without a `url`.

```bash
grep -R --line-number 'remote "gdrive"' .dvc ~/.config/dvc /etc/xdg/dvc 2>/dev/null || true
```

### 9.2 “Unable to acquire lock”

Cause:

* Another `dvc` process is still running or a previous run was interrupted.

Check running processes:

```bash
ps aux | grep -E "[d]vc|[p]ydrive|[g]drive" || true
```

If old processes exist, kill them and remove lock:

```bash
kill <PID1> <PID2> || true
kill -9 <PID1> <PID2> || true

rm -f .dvc/lock
rm -rf .dvc/tmp/lock* .dvc/tmp/rwlock* 2>/dev/null || true
```

### 9.3 Windows PowerShell errors (“touch”, “nano” not found)

Use:

* `notepad oauth.json`
* `Set-Content` or `[System.IO.File]::WriteAllText(...)`

### 9.4 Windows JSONDecodeError (BOM)

Fix by writing JSON without BOM:

```powershell
[System.IO.File]::WriteAllText("oauth.json", $json, (New-Object System.Text.UTF8Encoding($false)))
```

### 9.5 gcloud scp permission error on Windows

Use Cloud Shell upload + scp (Section 7.1), or fix IAM roles for your Windows account.

---

## 10) Security / Operational Notes

* Never commit:

  * `~/.gdrive/myremote-credentials.json`
  * `.dvc/config.local` (usually local-only)
* If secrets were exposed, rotate `client_secret` and regenerate `creds.json`.
* For CI/CD, consider switching to a Service Account (requires sharing the Drive folder with the SA email).

---

## Appendix: Useful commands

Show DVC config values:

```bash
dvc config --list
```

Show DVC environment:

```bash
dvc doctor
```

List instance zone (VM side):

```bash
gcloud compute instances list --project <PROJECT_ID> | grep <VM_NAME>
```

````

---

## Add a short link in `README.md`
Add something like this under “Data / Storage”:

```markdown
### Data versioning (DVC)
- Google Drive remote setup guide: `docs/dvc/gdrive-remote-setup.md`
````

---