#!/bin/bash
#
# Show harvesting progress status
#

LOG_DIR="/home/hilakese/role_based_segregation_dev/harvesting_data/logs"
DATA_DIR="/home/hilakese/role_based_segregation_dev/INPUT/UpdatatedHouseData"
TOKEN_FILE="/home/hilakese/role_based_segregation_dev/INPUT/id_token.csv"

# Counts
TOTAL=$(tail -n +2 "$TOKEN_FILE" | wc -l)
CSV_COUNT=$(ls -1 "$DATA_DIR"/*.csv 2>/dev/null | wc -l)
DONE_COUNT=$(grep -l "Done!" "$LOG_DIR"/harvest_*.log 2>/dev/null | wc -l)
RUNNING=$(squeue -u $USER -h 2>/dev/null | wc -l)

# Calculate
NEED_RETRY=$((TOTAL - DONE_COUNT))
WITH_DATA=$CSV_COUNT
WITHOUT_DATA=$((DONE_COUNT - CSV_COUNT))
if [ $WITHOUT_DATA -lt 0 ]; then WITHOUT_DATA=0; fi

# Progress bar
PCT=$((DONE_COUNT * 100 / TOTAL))
BAR_WIDTH=40
FILLED=$((PCT * BAR_WIDTH / 100))
EMPTY=$((BAR_WIDTH - FILLED))

echo ""
echo "=========================================="
echo "       HARVESTING PROGRESS STATUS"
echo "=========================================="
echo ""

# Progress bar
printf "  ["
printf "%${FILLED}s" | tr ' ' '█'
printf "%${EMPTY}s" | tr ' ' '░'
printf "] %3d%%\n" $PCT

echo ""
echo "  Total houses:      $TOTAL"
echo "  ─────────────────────────"
echo "  ✓ Completed:       $DONE_COUNT"
echo "    - With data:     $WITH_DATA"
echo "    - Empty (no data): $WITHOUT_DATA"
echo "  ⏳ Running:         $RUNNING"
echo "  ✗ Need retry:      $NEED_RETRY"
echo ""

# Show houses that need retry (no Done! in log)
if [ "$1" == "-v" ] || [ "$1" == "--verbose" ]; then
    echo "Houses needing retry:"
    echo "─────────────────────"
    tail -n +2 "$TOKEN_FILE" | cut -d',' -f1 | nl | while read idx id; do
        has_done=$(grep -l "House ID: *$id" "$LOG_DIR"/*.log 2>/dev/null | xargs -I{} tail -1 "{}" 2>/dev/null | grep -c "Done!" || echo 0)
        if [ "$has_done" -eq 0 ]; then
            printf "  %3d: %s\n" "$idx" "$id"
        fi
    done
    echo ""
fi

echo "Usage: $0 [-v|--verbose] to see houses needing retry"
echo ""
