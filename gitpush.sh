#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Please type some message to commit."
    exit
fi

git add . && git commit -m $1 && git push -u origin master