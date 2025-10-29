````{"variant":"standard","title":"Maya AI & Bridge – GCP 24/7 Deployment Documentation","id":"59732"}
# 🏥 Maya AI Receptionist – GCP Deployment Guide

This document explains how **Maya (AI receptionist)** and the **Bridge (audio handler)** are hosted on a single **Google Cloud Debian instance**, configured to run 24/7 automatically using `systemd`.

---

## 📦 Overview

### Components

| Component | Description | Path | Python Used |
|------------|--------------|------|--------------|
| **bridgez.py** | SIP/WebSocket Audio Bridge | `/home/apple/bridge-prod/bridgez.py` | `/usr/bin/python3` (system Python) |
| **appz.py** | Main Maya AI App (voice logic, Gemini/OpenAI, TTS/STT) | `/home/apple/shahid/appz.py` | `/home/apple/shahid/voxbay/bin/python3` (venv) |

Both are controlled by **systemd services**, ensuring automatic startup, crash recovery, and continuous operation.

---

## ☁️ GCP Environment

- **Instance name:** `maya-instance`
- **Zone:** `asia-south1-b`
- **User:** `apple`
- **OS:** Debian GNU/Linux 12 (kernel 6.1)
- **Python Virtual Environment (for appz):** `/home/apple/shahid/voxbay/`
- **System Python (for bridgez):** `/usr/bin/python3`

---

## ⚙️ 1. 24/7 Autostart Configuration (Systemd)

Two service units manage both processes permanently.

### `/etc/systemd/system/bridgez.service`

```ini
[Unit]
Description=Maya Bridge Service (System Python)
After=network.target

[Service]
User=apple
WorkingDirectory=/home/apple/bridge-prod
ExecStart=/usr/bin/python3 /home/apple/bridge-prod/bridgez.py
Restart=always
RestartSec=3
StandardOutput=append:/home/apple/bridge-prod/bridgez.log
StandardError=append:/home/apple/bridge-prod/bridgez-error.log

[Install]
WantedBy=multi-user.target
```

### `/etc/systemd/system/mayaz.service`

```ini
[Unit]
Description=Maya Main App Service (Voxbay venv)
After=bridgez.service
Requires=bridgez.service

[Service]
Type=simple
User=apple
WorkingDirectory=/home/apple/shahid
ExecStart=/home/apple/shahid/voxbay/bin/python3 /home/apple/shahid/appz.py
Restart=always
RestartSec=5
StandardOutput=append:/home/apple/shahid/appz.log
StandardError=append:/home/apple/shahid/appz-error.log

[Install]
WantedBy=multi-user.target
```

---

## ⚡ 2. Enabling and Starting the Services

After creating both service files:

```bash
sudo systemctl daemon-reload
sudo systemctl enable bridgez.service
sudo systemctl enable mayaz.service
sudo systemctl start bridgez.service
sudo systemctl start mayaz.service
```

---

## 🧭 3. How It Runs 24/7

- Both services are managed by **systemd**.  
- They **auto-start at boot**.  
- They **auto-restart on crash** (`Restart=always`).  
- Logs are written to:
  - `/home/apple/bridge-prod/bridgez.log`
  - `/home/apple/shahid/appz.log`
- They run under user `apple` and restart automatically if the instance reboots or loses network.

---

## 🧰 4. Management Commands

### 🔄 Restart both services manually

```bash
sudo systemctl restart bridgez.service
sudo systemctl restart mayaz.service
```

### 🔍 Check service status

```bash
sudo systemctl status bridgez.service
sudo systemctl status mayaz.service
```

### 📜 View live logs

```bash
journalctl -u bridgez.service -f
journalctl -u mayaz.service -f
```

### 💾 Or view log files directly

```bash
tail -f /home/apple/bridge-prod/bridgez.log
tail -f /home/apple/shahid/appz.log
```

---

## 🧮 5. Stop or Pause Services

### Stop both services
```bash
sudo systemctl stop bridgez.service
sudo systemctl stop mayaz.service
```

### Disable autostart on reboot
```bash
sudo systemctl disable bridgez.service
sudo systemctl disable mayaz.service
```

---

## 🧹 6. Clean Uninstall (When You’re Done)

If you no longer need Maya on this instance:

1. **Stop the services**
   ```bash
   sudo systemctl stop bridgez.service
   sudo systemctl stop mayaz.service
   ```

2. **Disable and delete service files**
   ```bash
   sudo systemctl disable bridgez.service
   sudo systemctl disable mayaz.service
   sudo rm /etc/systemd/system/bridgez.service
   sudo rm /etc/systemd/system/mayaz.service
   sudo systemctl daemon-reload
   ```

3. **Remove app directories (optional)**
   ```bash
   rm -rf /home/apple/bridge-prod
   rm -rf /home/apple/shahid
   ```

4. **(Optional)** Delete logs
   ```bash
   rm -f /home/apple/bridge-prod/*.log
   rm -f /home/apple/shahid/*.log
   ```

5. **(Optional)** Delete instance
   ```bash
   gcloud compute instances delete maya-instance --zone=asia-south1-b
   ```

---

## 🧩 7. Quick Restart Script

To make restarts easy, create a helper script:

`/home/apple/restart-maya.sh`
```bash
#!/bin/bash
sudo systemctl restart bridgez.service
sudo systemctl restart mayaz.service
sudo systemctl status bridgez.service --no-pager
sudo systemctl status mayaz.service --no-pager
```

Make it executable:
```bash
chmod +x /home/apple/restart-maya.sh
```

Run anytime:
```bash
./restart-maya.sh
```

---

## 🧠 8. Verification Checklist

| Check | Command | Expected |
|-------|----------|----------|
| Bridge running | `sudo systemctl status bridgez.service` | `active (running)` |
| Maya app running | `sudo systemctl status mayaz.service` | `active (running)` |
| Auto-start on reboot | `sudo reboot` → then check again | Both active |
| Port usage | `sudo lsof -i :8081` | Only one python3 process |
| Logs present | `tail bridgez.log` / `tail appz.log` | Recent activity |
