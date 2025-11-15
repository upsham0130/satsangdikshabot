# Deploying to Streamlit Cloud

Follow these steps to deploy your Streamlit app to Streamlit Cloud:

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at https://streamlit.io/cloud)

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `satsang-chatbot`)
3. **Don't** initialize with README, .gitignore, or license (we already have these)

## Step 2: Initialize Git and Push Your Code

If you haven't already initialized git in your project:

```bash
# Navigate to your project directory
cd "/home/upsham/Sypram/rag llm"

# Initialize git repository
git init

# Add all files
git add .

# Make your first commit
git commit -m "Initial commit: Satsang Diksha chatbot"

# Add your GitHub repository as remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Set Up Secrets for API Keys

**IMPORTANT**: Never commit API keys directly to your repository!

1. In your GitHub repository, go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add your Google API key:
   - Name: `GOOGLE_API_KEY`
   - Value: Your actual API key

## Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: Select your repository
   - **Branch**: `main` (or `master`)
   - **Main file path**: 
     - For Satsang app: `satsang_app.py`
     - For Inventory app: `app.py`
   - **App URL**: Choose a custom URL (optional)
5. Click **"Advanced settings"** and add your secret:
   - **Secrets**: 
     ```toml
     GOOGLE_API_KEY = "your-api-key-here"
     ```
   OR use the secrets manager in Streamlit Cloud UI
6. Click **"Deploy"**

## Step 5: Update Your App to Use Secrets

Update your app files to read from Streamlit secrets instead of hardcoded keys:

### For `satsang_app.py` and `app.py`:

Replace:
```python
os.environ["GOOGLE_API_KEY"] = "AIzaSyA32QT_Nb6f2-6NxG31ZMx6AxBwbrTOOIw"
```

With:
```python
# Try to get from Streamlit secrets, fallback to environment variable
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    pass  # Already set
else:
    st.error("⚠️ GOOGLE_API_KEY not found. Please set it in Streamlit Cloud secrets.")
    st.stop()
```

## Step 6: Handle PDF Files

Streamlit Cloud has limited storage. You have two options:

### Option A: Include PDF in Repository (Recommended for small files)
- Make sure your PDF file is committed to git
- The app will work as-is

### Option B: Use Streamlit File Uploader
- Modify the app to allow users to upload PDFs
- Store temporarily in session state

## Troubleshooting

### App won't deploy
- Check that `requirements.txt` is in the root directory
- Verify all dependencies are listed
- Check the deployment logs in Streamlit Cloud

### API Key errors
- Verify secrets are set correctly in Streamlit Cloud
- Check that the secret name matches exactly: `GOOGLE_API_KEY`

### PDF not found
- Ensure PDF file is committed to git
- Check the file path matches exactly (case-sensitive)

### Quota errors
- Your API quota applies to Streamlit Cloud too
- Consider implementing better rate limiting
- Or upgrade your Google API plan

## Updating Your App

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push
```

Streamlit Cloud will automatically redeploy your app!

## Multiple Apps

To deploy multiple apps from the same repository:

1. Deploy the first app (e.g., `satsang_app.py`)
2. Click **"New app"** again
3. Select the same repository
4. Change the **Main file path** to the other app (e.g., `app.py`)
5. Deploy

Each app gets its own URL!

