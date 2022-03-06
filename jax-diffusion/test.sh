set -e

trap 'printf \\a' EXIT

cat prompts.txt | \
  while read l; do
    nix develop -c chrt -b 0 python run.py "$l"
  done
