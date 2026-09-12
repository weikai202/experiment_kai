#!/bin/bash
set -e
umask 077
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /root/toolsandbox
exec /root/.local/bin/toolsandbox-with-secrets live-calibration
