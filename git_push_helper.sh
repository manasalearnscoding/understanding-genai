#!/bin/bash
# Git push helper script with authentication support

# Option 1: Use GitHub Personal Access Token from environment variable
if [ -n "$GITHUB_TOKEN" ]; then
    echo "Using GITHUB_TOKEN from environment..."
    git remote set-url origin https://${GITHUB_TOKEN}@github.com/manasalearnscoding/understanding-genai.git
    git push
    exit $?
fi

# Option 2: Use SSH (if SSH keys are set up)
if ssh -o ConnectTimeout=5 -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "SSH authentication available, using SSH..."
    git remote set-url origin git@github.com:manasalearnscoding/understanding-genai.git
    git push
    exit $?
fi

# Option 3: Interactive prompt for token
echo "GitHub authentication required."
echo "Please provide your GitHub Personal Access Token:"
echo "(Create one at: https://github.com/settings/tokens)"
read -s GITHUB_TOKEN

if [ -n "$GITHUB_TOKEN" ]; then
    git remote set-url origin https://${GITHUB_TOKEN}@github.com/manasalearnscoding/understanding-genai.git
    git push
    exit $?
else
    echo "No token provided. Cannot push."
    exit 1
fi

