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
if [ $? -eq 0 ]
then
  echo "Successfully push"
else 
  echo "Could not git push" >&2
  exit 1
fi
poetry build
if [ $? -eq 0 ]
then
  echo "Poetry build succeeded"
else
  echo "Poetry build failed" >&2
  exit 1
fi
poetry publish -u __token__
if [ $? -eq 0 ]
then
  echo "Poetry push succeeded"
else
  echo "Poetry push failed" >&2
  exit 1
fi


