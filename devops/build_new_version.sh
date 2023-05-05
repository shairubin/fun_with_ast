#!/bin/bash
echo "start build new version-patch"
poetry version patch
git commint -a 'new patch version update 
git push 
poetry build
