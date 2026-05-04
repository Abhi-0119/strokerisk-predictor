# Deployment Guide

Two free options. Pick one. Both end with a public URL you can share with the class.

---

## Option A — Streamlit Community Cloud (recommended)

**Time to deploy:** about 5 minutes. **Requires:** a free GitHub account.

### 1. Create a GitHub repo

1. Go to https://github.com/new
2. Repo name: `strokerisk-predictor` (or whatever you like)
3. Set it to **Public** (free Streamlit Cloud only deploys public repos)
4. Skip the README/.gitignore options (we have our own)
5. Click **Create repository**

### 2. Push these files

From your terminal, inside the `strokerisk_app` folder:

```bash
git init
git add .
git commit -m "Initial StrokeRisk Predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/strokerisk-predictor.git
git push -u origin main
```

If git asks for credentials, use a personal access token (GitHub → Settings → Developer settings → Personal access tokens → Generate new token, classic, with `repo` scope).

### 3. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **Create app** → **Deploy a public app from GitHub**
4. Pick your `strokerisk-predictor` repo
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy**

It will install the dependencies from `requirements.txt`, run `streamlit run app.py`, and give you a public URL like:

```
https://YOUR_USERNAME-strokerisk-predictor-app-xxxxx.streamlit.app
```

Share that URL with the class. Done.

### 4. Updating the app later

Edit any file, commit and push to GitHub, and Streamlit Cloud auto-redeploys within a minute.

---

## Option B — Hugging Face Spaces (no GitHub needed)

**Time to deploy:** about 5 minutes. **Requires:** a free Hugging Face account.

### 1. Create a Space

1. Go to https://huggingface.co/new-space
2. Space name: `strokerisk-predictor`
3. License: `mit`
4. SDK: **Streamlit**
5. Hardware: **CPU basic (free)**
6. Visibility: **Public**
7. Click **Create Space**

### 2. Upload the files

You'll see an empty Space repo with an "Add file" button.

Upload these (drag-and-drop in the web UI works fine):
- `app.py`
- `train_model.py`
- `stroke_model.joblib`
- `brainStrokeDataset.csv`
- `requirements.txt`
- `README.md`

(Don't upload `.gitignore` or `DEPLOY.md`. Skip `__pycache__` if you have it.)

### 3. Wait for the build

The Space rebuilds automatically. Watch the **Logs** tab. After about 2 minutes you'll see "Your app is now live".

Your URL will be:

```
https://huggingface.co/spaces/YOUR_USERNAME/strokerisk-predictor
```

Share that with the class.

---

## Sanity check before deploying

Run this locally first to make sure everything works:

```bash
cd strokerisk_app
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Try a high-risk profile (age 75, hypertension yes, glucose 200) and a low-risk profile (age 25, healthy values). The risk percentage should clearly differ between them.

If `stroke_model.joblib` is missing, run `python train_model.py` first.

---

## Keeping the app awake (free tier)

Streamlit Community Cloud puts free-tier apps to sleep after about 7 days of no traffic. When that happens, anyone visiting the URL sees a "Yes, get this app back up!" button. They click it, the app wakes up in 30-60 seconds, and works normally. So it's not actually broken — just napping.

If you want the app to stay awake permanently so the class never sees the wake-up screen, set up a free uptime monitor. Takes 2 minutes:

1. Go to https://uptimerobot.com and create a free account
2. Click **Add New Monitor**
3. Monitor Type: **HTTP(s)**
4. Friendly Name: `StrokeRisk Predictor`
5. URL: `https://strokerisk-predictor.streamlit.app`
6. Monitoring Interval: **5 minutes** (free tier minimum)
7. Click **Create Monitor**

UptimeRobot will hit your URL every 5 minutes from a global pool of servers. As long as something pings the app, Streamlit Cloud counts it as active and won't sleep it. Bonus: you'll get an email alert if the app actually goes down for some other reason.

Cron-Job.org and BetterStack are equally good free alternatives if you prefer those.

## Troubleshooting

**"Module not found" on deploy**: confirm `requirements.txt` is in the repo root, not nested.

**Big file warning on GitHub**: `stroke_model.joblib` is small (under 1 MB) so this should not happen. If it does, you can rebuild it from `train_model.py` on first run.

**App is "in the oven" forever on Streamlit Cloud**: hit the three-dot menu → Reboot.

**App says "This app has gone to sleep due to inactivity"**: that's the free-tier sleep. Click the wake-up button, or set up UptimeRobot using the section above.

**App boots but Calculate button errors out**: usually a sklearn/pandas version mismatch. The app retrains the model from the CSV at startup specifically to avoid this, so this should not happen. If it does, hit Reboot from the three-dot menu and check the logs tab.

**Hugging Face shows "Build error"**: check the build logs for a missing package. Pin the version in `requirements.txt` and retry.
