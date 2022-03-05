set -e

trap 'printf \\a' EXIT

cat prompts.txt | \
  while read l; do
    nix develop -c python run.py "$l"
  done
