# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:06:19 2016

@author: username
"""

import shapefile as sf, pylab

map_f = sf.Reader('tl_2010_us_state10/tl_2010_us_state10.shp')
state_metadata = map_f.records()
state_shapes = map_f.shapes()

for n in range(len(state_metadata)):
  pylab.plot([px[0] if px[0] <0 else px[0]-360 for px in state_shapes[n].points],[px[1] for px in state_shapes[n].points],'k.',ms=2)

for n in range(len(state_metadata)):
  pylab.plot(float(state_metadata[n][13]),float(state_metadata[n][12]),'o')
pylab.axis('scaled')