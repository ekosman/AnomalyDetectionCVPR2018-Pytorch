#!/bin/bash

# Usage info
help_str() {
cat << EOF
Create a conda environment and install adCVPR18 and its dependencies

Optional arguments:
    -h, --help      Display this help and exit.

Examples:

    ./install.sh
EOF
}


# switch to parent directory (no symlink)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

# parse arguments

# Create conda environment
ERR=$(conda create --name adCVPR18 --file environment.yml -c defaults -c pytorch -c conda-forge -y 3>&2 2>&1 1>&3 3>&- )
# check for user abort
if [[ $? -ne 0 || "$ERR" == *"CondaSystemExit: Exiting."* ]]; then
  >&2 echo $ERR
  exit 1
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh &&
conda activate adCVPR18 &&

cat << EOF

=== Installation finished. ===

To run the package, activate the environment:

conda activate adCVPR18

EOF
