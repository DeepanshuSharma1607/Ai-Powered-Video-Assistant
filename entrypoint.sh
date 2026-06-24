#!/bin/bash
set -e

# Ensure sessions.json exists as a valid JSON file (the bind-mount may
# have created it as an empty file or directory if it didn't exist yet).
if [ ! -s /app/sessions.json ]; then
    echo '{}' > /app/sessions.json
fi

exec "$@"