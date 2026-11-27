#!/bin/bash
# Fetch the needle-test story corpus (public-domain novels, Project Gutenberg).
# Plain prose must be coherent narrative (user rule 2026-10-05), not random
# filler. P&P ~380K words + Moby-Dick ~250K words ~ 500K tokens total: enough
# headroom for a 512K-token haystack, and we take slices from the front.
set -u
DIR="$(dirname "$0")/corpus"
mkdir -p "$DIR"
cd "$DIR" || exit 1
dl() { # $1=gutenberg id  $2=outfile
  [ -s "$2" ] && { echo "SKIP $2 (exists)"; return 0; }
  curl -s -m 300 -f -o "$2.tmp" "https://www.gutenberg.org/cache/epub/$1/pg$1.txt" \
    && mv "$2.tmp" "$2" && echo "OK $1 -> $2" \
    || { rm -f "$2.tmp"; echo "FAIL $1"; return 1; }
}
dl 1342 pride_and_prejudice.txt      # Austen
dl 2701    moby_dick.txt              # Melville
dl 1661    sherlock_holmes.txt         # Conan Doyle (short-story collection)
dl 1400    great_expectations.txt      # Dickens
for f in *.txt; do [ -f "$f" ] && echo "$f $(wc -w <"$f") words"; done
