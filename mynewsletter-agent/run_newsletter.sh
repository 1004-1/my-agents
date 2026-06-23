#!/bin/bash

cd /Users/soft1003/MyAgents/agents-in-github/mynewsletter-agent

source .venv/bin/activate

python -u main.py >> logs/newsletter.log 2>&1
