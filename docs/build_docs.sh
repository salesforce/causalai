#!/bin/bash
git checkout main
sphinx-build -b html docs/source docs/build/html
git add docs
git commit -m "build sphinx docs"
git push
rm -rf ../temp_docs
mkdir -p ../temp_docs
cp -R docs/build/html/* ../temp_docs/
git checkout gh-pages
rm -rf latest
mkdir -p latest
cp -R ../temp_docs/ latest/
git add latest
git commit -m "added latest docs"
git push
git checkout main
rm -rf ../temp_docs
