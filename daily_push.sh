#!/bin/bash
read -p "請輸入今天的 commit 訊息：" msg
git add .
git commit -m "$msg"
git push