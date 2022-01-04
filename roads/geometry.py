import math
import numpy as np



def fitLine(px, py):
	m, b = np.polyfit(px, py, 1)
	
	err = 0
	for x,y in zip(px, py):
		e =  m*x+b - y
		err += e*e

	err /= len(px)

	return err, m, b




def distanceBetweenPoints(p1,p2):
	d = p1-p2
	dx = d.real
	dy = d.imag
	return math.sqrt(dx*dx+dy*dy)


def distanceBetweenPointsSq(p1,p2):
	d = p1-p2
	dx = d.real
	dy = d.imag
	return dx*dx+dy*dy


def dotProduct(v1,v2):
	return v1.real*v2.real + v1.imag*v2.imag


def crossProduct(v1,v2):
	return v1.real*v2.imag - v1.imag*v2.real


def normalized(p):
	d = abs(p)
	return p/d



def threePointCurvature(p0, p1, p2):
	return crossProduct(p2-p1, p0-p1) / abs(p2-p0)


def centerOfCircle(z1,z2,z3):
	a = 1j*(z1-z2)
	b = 1j*(z3-z2)
	if a.real:
		m1 = a.imag/a.real
		c = (z1-z2)/2
		p1 = z2+c
		b1 = p1.imag-m1*p1.real
	if b.real:
		m2 = b.imag/b.real
		d = (z3-z2)/2
		p2 = z2+d
		b2 = p2.imag-m2*p2.real
	if a.real and b.real:
		x = (b2-b1)/(m1-m2)
		y = (m2*b1-m1*b2)/(m2-m1)
	elif a.real:
		x,y = 0,b1
	elif b.real:
		x,y = 0,b2
	else:
		x,y = 0,0
	center = x+1j*y
	radius = abs(center-z1)
	return complex(x,y),radius


def averageAngles(*angles):
	"""Average (mean) of angles

	Return the average of an input sequence of angles. The result is between
	``0`` and ``2 * math.pi``.
	If the average is not defined (e.g. ``average_angles([0, math.pi]))``,
	a ``ValueError`` is raised.
	"""

	x = sum(math.cos(a) for a in angles)
	y = sum(math.sin(a) for a in angles)

	if x == 0 and y == 0:
		raise ValueError(
			"The angle average of the inputs is undefined: %r" % angles)

	# To get outputs from -pi to +pi, delete everything but math.atan2() here.
	r = math.fmod(math.atan2(y, x) + 2 * math.pi, 2 * math.pi)
	if r > math.pi:
		r -= 2*math.pi
	return r


def subtractAngles(lhs, rhs):
	"""Return the signed difference between angles lhs and rhs

	Return ``(lhs - rhs)``, the value will be within ``[-math.pi, math.pi)``.
	Both ``lhs`` and ``rhs`` may either be zero-based (within
	``[0, 2*math.pi]``), or ``-pi``-based (within ``[-math.pi, math.pi]``).
	"""

	return math.fmod((lhs - rhs) + math.pi * 3, 2 * math.pi) - math.pi



def isLine(path, t0, t1, line_threshold = 10):
	plen = path.length(t0, t1)

	p0 = path.point(t0)
	p1 = path.point(t1)
	pm = path.point((t0+t1)*0.5)

	# try to fit a circle	
	try:
		center, radius = centerOfCircle(p0,p1,pm)
	except:
		return True

	n = radius / plen

	return n > line_threshold

