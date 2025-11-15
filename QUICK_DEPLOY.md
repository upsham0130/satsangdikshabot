# Quick Deploy Guide to Streamlit Cloud

## 🚀 Fast Track (5 minutes)

### 1. Push to GitHub
```bash
cd "/home/upsham/Sypram/rag llm"
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set **Main file path**: `satsang_app.py` (or `app.py`)
6. Click **"Advanced settings"** → **"Secrets"**
7. Add:
   ```toml
   GOOGLE_API_KEY = "your-actual-api-key-here"
   ```
8. Click **"Deploy"**

### 3. Done! 🎉
Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

## 📝 Important Notes

- ✅ Your PDF files will be included if they're in the repo
- ✅ API key is stored securely in Streamlit secrets
- ✅ App auto-updates when you push to GitHub
- ⚠️ Free tier has resource limits
- ⚠️ API quota still applies (same as local)

## 🔄 Updating Your App

Just push to GitHub:
```bash
git add .
git commit -m "Update app"
git push
```

Streamlit Cloud redeploys automatically!

