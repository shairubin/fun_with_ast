#!/bin/bash
echo "start build new version-patch"
echo "add shh key"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
echo "patch a version"
poetry version patch
echo "adding commit message"
git commit -m 'new patch version update'
echo "push to git"
git push
echo "poetry build"
poetry build
