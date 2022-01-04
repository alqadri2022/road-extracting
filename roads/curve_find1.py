import math
import re
import numpy as np
from pyearth import Earth
#from scipy.optimize import minimize
from .geometry import isLine, normalized, crossProduct
from .curves import Spiral,Line,Circle,prepareCurves


def findEarthCorners(model):
	feature_corners = set([0,1])
	for b in model.basis_:
		if b.is_pruned(): continue
		for mo in re.finditer("([a-z]+[0-9]*)?([-+]?[0-9.]+(?:[eE][-]?[0-9]+)?)([-+][a-z]+[0-9]*)?",str(b)):
			v = float(mo.group(2))
			if mo.group(1):
				v = -v
			feature_corners.add(v)
	feature_corners = list(feature_corners)
	feature_corners.sort()


	X = np.reshape(feature_corners, (-1,1))
	Y = model.predict(X)

	return [complex(x,y) for x,y in zip(feature_corners,Y)]	


# remove the most redundant corner (the one that lies mostly on a line)
def removeCorner(corners):
	index = []
	for i in range(1, len(corners)-1):
		d1 = corners[i-1] - corners[i]
		d2 = corners[i+1] - corners[i]

		s = (d2.real - d1.real) / 2

		v1 = normalized(complex(d1.real, d1.imag*s))
		v2 = normalized(complex(d2.real, d2.imag*s))
		
		c = abs(crossProduct(v1,v2))
		index.append((c,i))

	index.sort()
	del corners[index[0][1]]
	


def extractFeaturesAlg(path, samples, use_curvature = False):
	if use_curvature:
		points = path.samples(samples, sample_fn="curvature")
	else:
		points = path.samples(samples, sample_fn="bearing-change")

	model = Earth()#max_terms = 4005)

	X = np.real(points)
	Y = np.imag(points)

	X = X.reshape(-1,1)

	model.fit(X,Y)

	corners = findEarthCorners(model)

	# make the number of corners even
	if len(corners) &1 == 1: 
		removeCorner(corners)

	return corners


def quantizeCorners(feature_corners):
	tot_lines = len(feature_corners)-1
	min_slope = 0.21

	def proc(i):
		p0 = feature_corners[i]
		p1 = feature_corners[i+1]

		sin_slope = math.atan2((p1.imag - p0.imag)*100, p1.real - p0.real)

		if math.fabs(sin_slope) < min_slope:
			y = (p0.imag + p1.imag) * 0.5
			feature_corners[i]   = complex(feature_corners[i].real, y)
			feature_corners[i+1] = complex(feature_corners[i+1].real, y)

	change = True
	while change:
		change = False
		for i in range(0,tot_lines,2): 
			change = proc(i) or change
		for i in range(1,tot_lines,2):
			change = proc(i) or change

	return feature_corners
#def isSpiral(path, p0, p1, spiral_threshold = 0.8):
#	return abs(p0.imag - p1.imag) > spiral_threshold * abs(p0.imag + p1.imag)


def buildCurvesFromCurvature(path, feature_corners):
	tot_curves = len(feature_corners)-1
	# there must be an odd number of curves. If that's not the case,
	# we add another one

	# there must be an odd number of curves
	assert(tot_curves&1) 

	max_curvature_for_line = 0.1

	# this is to make the curvature comparisons
	# scale invariant

	max_curvature_for_line /= path.size_scale

	curves = []

	for i in range(tot_curves):
		p0 = feature_corners[i]
		p1 = feature_corners[i+1]

		c = None

		if isSpiral(path, p0, p1):
			# fit a spiral
			c = Spiral(path, p0.real, p1.real)
			print("spiral")
		elif isLine(path, p0.real, p1.real):
			print("line")
			# fit a line
			c = Line(path, p0.real, p1.real) 
		else:
			print("circle")
			# fit a circle
			c = Circle(path, p0.real, p1.real)
		if c:
			curves.append(c)

	prepareCurves(curves)	

	return curves

#----------------------------------------
def buildCurves(path, plot_curvature_axis=None, curves=None):
	# "features" is a list of points that describe the
	# estimated curvature along the path
	feature_corners = extractFeaturesGeom(path)

	if plot_curvature_axis:
		X = list(np.real(feature_corners))
		Y = list(np.imag(feature_corners))
		plot_curvature_axis.plot(X, Y)

	if not curves:
		curves = buildCurvesFromCurvature(path, feature_corners)
