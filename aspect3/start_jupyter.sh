#!/bin/bash
# Starts a Jupyter notebook server on a GPU compute node.
# Run this on the login node, from inside aspect3/:
#   bash start_jupyter.sh
#
# Then on your LOCAL machine, run the ssh tunnel command printed below.
# Then in VSCode: "Select Kernel" → "Existing Jupyter Server" → paste the URL.

PORT=8899
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"   # absolute path to aspect3/

srun \
  --gres=gpu:1 \
  --mem=32G \
  --time=04:00:00 \
  --partition=all \
  --pty bash -c "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate structml1
    cd '${DIR}'
    NODE=\$(hostname)
    echo ''
    echo '============================================================'
    echo 'Jupyter starting on node: '\$NODE
    echo ''
    echo 'On your LOCAL machine run:'
    echo \"  ssh -L ${PORT}:\${NODE}:${PORT} \${USER}@\$(hostname -f | sed 's/\${NODE}\.//')\"
    echo ''
    echo 'Then in VSCode: Select Kernel → Existing Jupyter Server'
    echo 'URL will be printed below (starts with http://127.0.0.1:${PORT}/...)'
    echo '============================================================'
    echo ''
    jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0
  "
