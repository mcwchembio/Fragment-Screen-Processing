#!/bin/csh

basicFT2.com -in test.fid -out test.ft2 \
 -xP0 -73 -xP1 0 -xEXTX1 13ppm -xEXTXN 5ppm \
 -yP0 -90 -yP1 180 -yZFARG zf=2,auto 

sethdr test.ft2 -title mask

set istMaxRes = (`specStat.com -in test.ft2 -stat istMaxRes -brief`)

ist2D.com -in test.fid -istMaxRes $istMaxRes -mask mask.fid -out ist.ft2 \
 -xBASEARG POLY,auto \
 -xP0 -73 -xP1 0 -xEXTX1 13ppm -xEXTXN 5ppm \
 -yP0 -90 -yP1 180 -yZFARG zf=2,auto 

sethdr ist.ft2 -title NUS_25pct_IST
sleep 1
mv ./ist.ft2 ./test.ft2
sleep 1
