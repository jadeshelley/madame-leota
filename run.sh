#!/bin/bash

# Simple run script for Madame Leota
# This script activates the virtual environment and starts the application

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔮 Starting Madame Leota...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Please run install.sh first.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}.env file not found. Please copy .env.example to .env and configure it.${NC}"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start the application
python main.py 