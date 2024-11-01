set -e
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo "Building standalone classifier"

cd $SCRIPTPATH

make -j `nproc`

echo "Building standalone classifier OK"
