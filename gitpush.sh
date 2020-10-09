#!/bin/bash

# $# 는 인자의 개수
# $1 은 첫번째 인자를 문자열로
# $2 는 2번째 인자....

#   <if 문법>
# if [ ... ]; then
#    ...
# else
#    ...
# fi

if [ $# -eq 0 ]; then
    echo "Please type some message to commit."
    exit
fi

git add . && git commit -m "$@" && git push -u origin master