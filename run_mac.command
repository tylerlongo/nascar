#!/bin/bash
cd "$(dirname "$0")"

# Open the dashboard in the default browser after Flask has a moment to start.
(sleep 2 && open http://127.0.0.1:5000) &

# Start the NASCAR dashboard server.
python3 app.py

# Keep this window open if Python exits or shows an error.
echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
