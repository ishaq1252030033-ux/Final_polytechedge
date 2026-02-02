# PolytechEDGE

A web application for polytechnic students: college predictor, career guidance, job roadmap, study materials (K/I scheme), and admission info (poly & DSE).

## Features

- **College Predictor** – Get college recommendations based on scores, preferences, and category
- **Career Guidance** – Higher education and job-search guidance
- **Job Roadmap** – Step-by-step guidance for roles (e.g. full-stack developer)
- **Polytech Portal** – Study materials, question papers, notes by branch/semester
- **Admissions** – First-year polytechnic and DSE (diploma to B.Tech) admission info
- **Auth** – Sign up / login (email or Google/Apple OAuth)

## Tech Stack

- **Backend:** Python, Flask, Flask-SQLAlchemy
- **Frontend:** Bootstrap 5, Jinja2 templates
- **Data/ML:** Pandas, NumPy, Scikit-learn, PyPDF2
- **API:** OpenAI (optional), OAuth (Google, Apple)

---

## Local Development

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/polytechedge.git
cd polytechedge/edge
pip install -r requirements.txt
```

### 2. Environment variables

Copy the example env and set your values (never commit `.env`):

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- `SECRET_KEY` – random string for session security
- `OPENAI_API_KEY` – if using AI features
- Optional: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `APPLE_*` for OAuth

### 3. Run locally

```bash
python app.py
```

Open **http://127.0.0.1:5000/**

### 4. (Optional) Process data and train predictor

If you have DTE cutoff PDFs and want to retrain the predictor:

1. Put PDFs in the expected location (see project docs).
2. Visit **http://127.0.0.1:5000/process_data** and run the process.

---

## Deploy to GitHub

1. Create a new repo on GitHub (e.g. `polytechedge`).
2. In the project folder (e.g. `edge`):

```bash
git init
git add .
git commit -m "Initial commit: PolytechEDGE app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

- Do **not** push `.env` (it’s in `.gitignore`).
- Do push `.env.example` so others know which variables to set.

---

## Deploy to Render.com

### Option A: Using the Dashboard

1. Go to [Render](https://render.com) and sign in (e.g. with GitHub).
2. **New → Web Service**.
3. Connect the GitHub repo and select the **branch** (e.g. `main`).
4. **Root Directory:** set to `edge` if the app lives in an `edge` subfolder; otherwise leave blank.
5. **Runtime:** Python 3.
6. **Build command:**
   ```bash
   pip install -r requirements.txt
   ```
7. **Start command:**
   ```bash
   gunicorn --bind 0.0.0.0:$PORT app:app
   ```
8. **Environment:** Add variables (use “Add Environment Variable”):
   - `SECRET_KEY` – generate a long random string.
   - `OPENAI_API_KEY` – if you use AI features.
   - Optional: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `APPLE_CLIENT_ID`, `APPLE_CLIENT_SECRET`, `APPLE_REDIRECT_URI` for OAuth.  
   For OAuth redirect URI use your Render URL, e.g. `https://YOUR_SERVICE.onrender.com/auth/apple/callback`.
9. **Create Web Service.** Render will build and deploy. Your app will be at `https://YOUR_SERVICE.onrender.com`.

### Option B: Using `render.yaml` (Blueprint)

The repo includes a `render.yaml` that defines build and start commands. You can use it with **Blueprint** or copy the same commands into a new Web Service as in Option A.

### Database on Render

- **Default:** The app uses **SQLite** if `DATABASE_URL` is not set. On Render’s free tier the filesystem is ephemeral, so SQLite data is lost on redeploy or restart.
- **Persistent data:** Add a **Postgres** database in Render (Dashboard → New → PostgreSQL). Render sets `DATABASE_URL` automatically. The app already uses `DATABASE_URL` when present (and switches to Postgres).

### After first deploy

- If you use Postgres, ensure tables exist (the app calls `db.create_all()` where needed; you can also run a one-off command or a small script if you add one).
- Set OAuth redirect URIs in Google/Apple consoles to your Render URL (e.g. `https://YOUR_SERVICE.onrender.com/auth/google/callback`).

---

## Project Structure

```
edge/
├── app.py              # Main Flask app
├── requirements.txt     # Python dependencies
├── render.yaml          # Optional Render Blueprint
├── .env.example         # Example env vars (copy to .env)
├── utils/               # Helpers (prediction, college URLs, etc.)
├── templates/           # HTML templates
├── static/              # CSS, JS, images
├── models/              # Saved ML models (if any)
└── data/                # Processed data (if any)
```

## License

This project is open-source and available for educational use.
