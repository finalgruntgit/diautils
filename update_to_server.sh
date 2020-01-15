#!/bin/sh
./install_local.sh
git commit -am.
git pull
git commit -am.
git push
