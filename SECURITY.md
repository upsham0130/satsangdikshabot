# 🔒 API Key Security Guide

## ✅ What's Been Done

1. **Removed all hardcoded API keys** from your code
2. **Added `.env` to `.gitignore`** - your API keys won't be committed
3. **Updated all apps** to use environment variables
4. **Added `python-dotenv`** to load `.env` files

## 🚀 Quick Setup

### For Local Development:

1. **Create a `.env` file** in the project root:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** and add your API key:
   ```env
   GOOGLE_API_KEY=your-actual-api-key-here
   ```

3. **The `.env` file is already in `.gitignore`** - it won't be committed to GitHub!

### For Streamlit Cloud:

1. Go to your app on Streamlit Cloud
2. Click **"Settings"** → **"Secrets"**
3. Add:
   ```toml
   GOOGLE_API_KEY = "your-actual-api-key-here"
   ```

## ✅ Verify Your Keys Are Safe

### Check what will be committed:
```bash
git status
git diff
```

You should **NOT** see your API key in any files!

### Check `.gitignore`:
```bash
cat .gitignore | grep -E "(\.env|secrets)"
```

Should show:
- `.env`
- `.env.local`
- `.streamlit/secrets.toml`

## ⚠️ If You Already Committed Your API Key

If you accidentally committed your API key before, you need to:

1. **Rotate your API key** (get a new one from Google)
2. **Remove it from git history**:
   ```bash
   # Use git filter-branch or BFG Repo-Cleaner
   # Or simply create a new commit that removes it
   ```

3. **Force push** (⚠️ only if you're the only one using the repo):
   ```bash
   git push --force
   ```

## 📋 Checklist Before Pushing to GitHub

- [ ] No API keys in any `.py` files
- [ ] `.env` file exists locally but is NOT committed
- [ ] `.env.example` is committed (template only)
- [ ] `.gitignore` includes `.env` and `.streamlit/secrets.toml`
- [ ] Tested that app works with `.env` file
- [ ] Ready to add secrets in Streamlit Cloud

## 🔍 How to Check for Exposed Keys

```bash
# Search for potential API keys in your code
grep -r "AIzaSy" . --exclude-dir=.git
# Should return nothing (or only in .env which is ignored)

# Check git history (if you want to be thorough)
git log --all --source -- "*" | grep -i "AIzaSy"
# Should return nothing
```

## 🎯 Best Practices

1. ✅ **Always use environment variables** for secrets
2. ✅ **Never commit `.env` files**
3. ✅ **Use `.env.example`** as a template
4. ✅ **Rotate keys** if accidentally exposed
5. ✅ **Use different keys** for development and production
6. ✅ **Review commits** before pushing

