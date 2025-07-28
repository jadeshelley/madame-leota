#!/bin/bash

# Update script for Madame Leota
# This script pulls the latest changes from git and updates dependencies

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🔮 Updating Madame Leota...${NC}"

# Check if this is a git repository
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Warning: Not a git repository. Skipping git pull.${NC}"
else
    # Pull latest changes
    echo "Pulling latest changes from git..."
    git pull origin main
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    
    # Update Python dependencies
    echo "Updating Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo -e "${YELLOW}Virtual environment not found. Please run install.sh first.${NC}"
    exit 1
fi

# Create any new directories
mkdir -p assets/faces cache/audio logs

echo -e "${GREEN}✅ Update complete!${NC}"
echo "You can now restart Madame Leota with: ./run.sh" 