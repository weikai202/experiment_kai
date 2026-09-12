#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set +x
umask 077
cd /root/toolsandbox
exec /root/.local/bin/toolsandbox-with-secrets reflection-smoke
