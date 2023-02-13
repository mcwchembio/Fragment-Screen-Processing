#!/bin/csh

#
# Generated by basicFT2.com Version 2022.340.15.42
#
# basicFT2.com -in test.fid -out test.ft2 -xSOL POLY,time \
#              -xEXTX1 3% -xEXTXN 47% -xQ2 1 -xP0 -43.0 -xBASEARG POLY,auto,ord=4 \
#              -yQ2 1 -yP0 -90.0 -yP1 180.0 

nmrPipe -in test.fid \
| nmrPipe -fn POLY -time \
| nmrPipe -fn SP -off 0.5 -end 1 -pow 2 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn ZF -zf 1 -auto \
| nmrPipe -fn FT -verb \
| nmrPipe -fn PS -p0 -43.0 -p1 0.0 -di \
| nmrPipe -fn EXT -x1 3% -xn 47% -sw \
| nmrPipe -fn TP \
| nmrPipe -fn SP -off 0.5 -end 1 -pow 1 -elb 0.0 -glb 0.0 -c 1.0 \
| nmrPipe -fn ZF -zf 1 -auto \
| nmrPipe -fn FT -verb \
| nmrPipe -fn PS -p0 -90.0 -p1 180.0 -di \
| nmrPipe -fn TP \
| nmrPipe -fn POLY -auto -ord 4 -verb \
  -out test.ft2 -ov