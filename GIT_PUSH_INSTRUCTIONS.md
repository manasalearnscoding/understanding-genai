# Git Push Authentication - Solutions

## Issue
Git push fails because authentication is required but interactive prompts don't work in this environment.

## Solutions (Choose One)

### Solution 1: Use GitHub Personal Access Token (Recommended)

1. **Create a token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo` (full control of private repositories)
   - Copy the token

2. **Use the helper script:**
   ```bash
   export GITHUB_TOKEN=your_token_here
   ./git_push_helper.sh
   ```

   OR manually:
   ```bash
   git remote set-url origin https://YOUR_TOKEN@github.com/manasalearnscoding/understanding-genai.git
   git push
   ```

### Solution 2: Set up SSH Keys

1. **Generate SSH key (if you don't have one):**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **Add to SSH agent:**
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. **Add public key to GitHub:**
   - Copy: `cat ~/.ssh/id_ed25519.pub`
   - Add at: https://github.com/settings/keys

4. **Use SSH remote:**
   ```bash
   git remote set-url origin git@github.com:manasalearnscoding/understanding-genai.git
   git push
   ```

### Solution 3: Use Git Credential Manager

```bash
git config --global credential.helper manager
git push
# Will prompt for credentials once, then save them
```

## Current Status

✅ **Fixed:**
- Added `.jsonl` result files to `.gitignore`
- Created `git_push_helper.sh` script
- Configured credential store

📝 **Files ready to commit:**
- `util/run_winobias_tracing.py` (modified)
- `util/scoring_and_patching_utils.py` (modified)
- `example_vocab_component_analysis.py` (new)
- `paired_analysis_guide.md` (new)
- `test_vocab_projection.py` (new)
- `.gitignore` (modified)

## Quick Push (if you have token)

```bash
export GITHUB_TOKEN=your_token_here
git remote set-url origin https://${GITHUB_TOKEN}@github.com/manasalearnscoding/understanding-genai.git
git add .
git commit -m "Add vocabulary projection pipeline for RQ1"
git push
```

