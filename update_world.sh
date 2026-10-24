#!/bin/bash
cd /home/aiuser/Desktop/voxel-world

# Run your AI agent to generate the next chunk/state update
python3 agent.py

# Check if there are any changes to commit
if [[ -n $(git status -s) ]]; then
    git add .
    git commit -m "Auto-expansion: Pi agent generated new voxel world state"
    git push origin main
fi
