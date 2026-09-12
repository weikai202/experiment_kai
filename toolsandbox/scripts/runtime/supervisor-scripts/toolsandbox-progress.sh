#!/bin/bash
set -e
umask 077
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
exec /usr/bin/python3 /root/toolsandbox-runtime/progress_monitor.py --campaign /root/toolsandbox-runtime/full-runs/full-live-20260912T204856-a397cf --online /root/toolsandbox-runtime/runs/live-calibration-20260912T204857-f74e7f
