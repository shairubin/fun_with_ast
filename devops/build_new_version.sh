#!/bin/bash
echo "start build new version-patch"
echo "add shh key"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
echo "patch a version"
poetry version patch
echo "adding commit message"
git add pyproject.toml
git commit -m 'new patch version update'
echo "push to git. Enter user name and your access token"
git push origin main
echo "poetry build"
poetry build
