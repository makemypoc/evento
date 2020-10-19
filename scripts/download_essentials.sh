#!/bin/bash

# downloading pre-compiled binary of rtsp simple server for emulating IP cam using ffmpeg
mkdir bin
cd bin

wget -c https://github.com/aler9/rtsp-simple-server/releases/download/v0.9.15/rtsp-simple-server_v0.9.15_linux_amd64.tar.gz -O - | tar -xz
rm rtsp-simple-server_v0.9.15_linux_amd64.tar.gz

# downloading ppyolo model
cd ../models

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hY9bISdxFF2V8i7akMV42f6nlC3b03F1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hY9bISdxFF2V8i7akMV42f6nlC3b03F1" -O ppyolo.tar.xz && rm -rf /tmp/cookies.txt | tar -xf
rm ppyolo.tar.xz
rm uc?*
