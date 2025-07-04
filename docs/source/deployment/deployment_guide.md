# Arkastone Deployment & Development Reference

This document serves as a full reference guide to deploy, maintain, and develop the Arkastone simulation system. It includes frontend (Streamlit), backend (FastAPI), and client (executable).

---

## 🔧 1. Local Setup Overview

### ✅ Folder Structure
```
arkastone/
├── streamlit_app_online.py      # Frontend
├── local_client.py              # Client
├── main.py                      # Backend (for Render)
├── downloads/client.exe         # Windows executable
├── assets/                      # Images like logos
├── requirements.txt             # Shared dependencies
├── .venv/                       # Virtual environment
```

---

## ☁️ 2. Backend Deployment (Render)

### ✅ Steps:
1. Push `main.py` and `requirements.txt` to a GitHub repo (e.g., `arkastone-backend`)
2. Go to [https://render.com](https://render.com)
3. Create a new **Web Service**
4. Connect GitHub repo
5. Configure:
    - Environment: Python
    - Start command: `uvicorn main:app --host 0.0.0.0 --port 10000`
    - Exposed Port: Render auto-detects (443)
6. Deploy and copy the domain (e.g., `https://arkastone-backend.onrender.com`)

---

## 💻 3. EC2 Setup for Frontend (Streamlit)

### ✅ EC2 Launch Checklist:
- Instance type: `t2.micro` (Free Tier)
- AMI: Ubuntu 22.04
- Security group:
    - Port 22 (SSH)
    - Port 80 (HTTP)
    - Port 443 (HTTPS)
    - Port 8501 (optional, for direct Streamlit access)
- Key pair: download `.pem` file and store safely
- Assign A records:
    - `@` → EC2 public IP
    - `www` → EC2 public IP

### ✅ SSH Into EC2:
```bash
chmod 400 arkastone-key.pem
ssh -i arkastone-aws-key.pem ubuntu@18.221.4.137
```

---

## ⚙️ 4. Streamlit Frontend Deployment

### ✅ Clone Frontend Repo:
```bash
git clone https://github.com/yourusername/arkastone-frontend.git
cd arkastone-frontend
```

### ✅ Install Environment:
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx certbot python3-certbot-nginx -y

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ✅ Start App in Background (tmux recommended):
```bash
sudo apt install tmux
tmux new -s streamlit
source .venv/bin/activate
streamlit run streamlit_app_online.py
# Ctrl+B then D to detach
```

---

## 🌐 5. NGINX + HTTPS

### ✅ NGINX Config (`/etc/nginx/sites-available/arkastone`)
```nginx
server {
    listen 80;
    server_name arkast.one www.arkast.one;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/arkastone /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### ✅ Enable HTTPS with Certbot:
```bash
sudo certbot --nginx -d arkast.one -d www.arkast.one
```

---

## 🧊 6. Building the Windows Executable

### ✅ Using PyInstaller:
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole local_client.py
```

Executable will be at `dist/local_client.exe`. Move it to `downloads/` for UI download.

---

## 🧪 7. Testing Flow

- Submit config via `arkast.one`
- Run client `.exe` locally
- Results stream back to Streamlit
- Backend receives updates via `/update_progress` and `/submit_result`

---

## 🚧 8. Under Development Banner

Add this to Streamlit:
```python
st.warning("🚧 This site is under active development. Some features may not work as expected.")
```

---

## 📌 9. Notes

- Use `BACKEND_URL = "https://arkastone-backend.onrender.com"` in both client and frontend
- Always test DNS propagation after domain changes (`ping arkast.one`)
- Use `tmux` or `systemd` to keep apps running
- Secure `.pem` files — if lost, you'll lose SSH access

---

## 🔁 10. Renewal

Test auto-renew for SSL:
```bash
sudo certbot renew --dry-run
```

---

## 📎 Useful Links

- AWS EC2: https://console.aws.amazon.com/ec2
- Render: https://dashboard.render.com
- Certbot: https://certbot.eff.org
- Streamlit: https://docs.streamlit.io
- Namecheap DNS: https://ap.www.namecheap.com/domains/list

---

## 👨‍💻 Maintained by: Furkan Ercan
