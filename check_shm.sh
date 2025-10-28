#!/bin/bash
# Check shared memory size and usage

echo "=== Shared Memory Status ==="
df -h /dev/shm 2>/dev/null || echo "❌ /dev/shm not found (might be using tmpfs)"

echo ""
echo "=== System Memory ==="
free -h 2>/dev/null || vmstat -s 2>/dev/null | head -5

echo ""
echo "=== Recommendation ==="
SHM_SIZE=$(df -k /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
if [ ! -z "$SHM_SIZE" ] && [ "$SHM_SIZE" -gt 2000000 ]; then
    echo "✅ You have enough shared memory ($(df -h /dev/shm | awk 'NR==2 {print $2}'))"
    echo "   You can use WORKERS=2 or WORKERS=4 for faster training"
    echo ""
    echo "   Update your .env:"
    echo "   WORKERS=2"
else
    echo "⚠️  Limited shared memory detected"
    echo "   Keep WORKERS=0 for stability"
fi
