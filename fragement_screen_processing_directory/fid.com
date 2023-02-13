#!/bin/csh

bruk2pipe -verb -in ./ser \
  -bad 0.0 -ext -aswap -AMX -decim 2000 -dspfvs 20 -grpdly 67.9862518310547  \
  -xN              1024  -yN               256  \
  -xT               512  -yT               128  \
  -xMODE            DQD  -yMODE        Complex  \
  -xSW        10000.000  -ySW         1946.283  \
  -xOBS         600.133  -yOBS          60.818  \
  -xCAR           4.780  -yCAR         120.000  \
  -xLAB              1H  -yLAB             15N  \
  -ndim               2  -aq2D         Complex  \
| nmrPipe -fn MULT -c 9.76562e-01 \
  -out ./test.fid -ov

#
# nmrDraw -process -in test.fid -fid test.fid


