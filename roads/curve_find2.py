from collections import namedtuple
import numpy as np
from .geometry import *
from .algorithms import getLargestSpans, getOverlaps, takeLowestIndex, findRuns
from .curves import Spiral, Line, Circle, prepareCurves


#distance_from_path_penalty = 2 
#spiral_distance_from_path_penalty = 0.1 #0.1
#tangent_mismatch_penalty = 1
#curvature_discontinuity_penalty = 50



def sampleCurvature(path, t0, t1):
	t0 = max(0,t0)
	t1 = min(1,t1)
	px = list(range_t(20, t0, t1, shrink=0.99))
	py = [path.curvature(x) for x in px]
	return px,py


def samplePosition(path, t0, t1):
	t0 = max(0,t0)
	t1 = min(1,t1)
	pt = [path.point(t) for t in range_t(20, t0, t1, shrink=0.99)]
	return np.real(pt), np.imag(pt)


#----------------------------------------------------------------

IndexedSpan = namedtuple("IndexedSpan", ("error", "range", "id"))

# check how can we merge pieces into larger 
# checker returns a value of how well a sequence can be joined into a single curve
def pickRanges(path, from_seg, to_seg, error_fn, threshold):

	#print("="*50)
	#print("Pick ranges:")
	segments = list(range(from_seg, to_seg))

	# slist is segments
	checker = lambda a, b: error_fn(path, segments[a], segments[b]) < threshold

	spans = list(getLargestSpans(len(segments), checker))

	print("Spans: ",spans)

	intersecting_index, isolated = getOverlaps(spans)
	# isolated spans are good, they can be output as they are

	# second number is fit error. It's 0 because it's a single segment
	isolated = list((rn,0) for rn in isolated)

	print("Isolated=",isolated)

	print("Intersections:")
	for k,v in intersecting_index.items():
		vs = ", ".join(str(a) for a in v)
		print(f"\t#{k} -> #{vs}")

	

	print("From seg= ",from_seg)

	if intersecting_index:
		span_index = {}
		for s in intersecting_index.keys():
			# span_id -> (fitness, span_range, span_id)
			# initially the range and the id are the same, but the range can change if it
			# gets trimmed if an intersecting span gets picked before
			print("Span ",s[0]+from_seg, s[1]+from_seg)
			span_index[s] = IndexedSpan(error_fn(path, s[0]+from_seg, s[1]+from_seg), s, s)

		# take the best span from the index, remove it,
		# update intersected and emit those no longer intersecting
		while span_index:

			#print("Span index:")
			#for k,v in span_index.items():
				#print(f"\t#{k} -> #{v}")

			spans = list(span_index.values())

			best_span_num = takeLowestIndex(spans)

			picked = spans[best_span_num]

			#print(f"Picked: #{picked.id}")

			isolated.append((picked.range, picked.error))

			x,y = picked.range

			# remove the range from intersecting ranges
			for iseg in intersecting_index[picked.id]:
				if iseg not in span_index: continue # this span was already removed
				a,b = span_index[iseg].range
				#if a,b is completely contained in x,y, just remove it
				if a>=x and b<=y:
					#print(f"Removed #{iseg} to ({a},{b})")
					del span_index[iseg]
					continue
				# see which side we trim it from
				# a,b -> range of span we're trimming
				# x,y -> range of span we removed
				if y >=b:  # trimming from the right
					if b>x:
						b = x
				elif x <= a:
					if a<y:
						a = y
				else: 
					raise Exception("Cutting segment cannot be strict subset of cut segment")

				#print(f"Changing ranges of #{iseg} : ({a},{b})")
				span_index[iseg] = IndexedSpan(error_fn(path, a+from_seg, b+from_seg), (a,b), iseg)

			# remove the chosen segment from the index
			del span_index[picked.id]

	#print("Ranges: ",isolated)
	#print("-"*50)
	return isolated # list of ((range start, range end), fit error)
		

# line_detect_thd : higher values -> lines need to be more straight  lower values -> slight curves taken as lines
# spiral_curve_slope_thd: higher values -> more curves taken as arcs  lower value -> curves more likely to be spieals
# line_merge_thd : higher value -> 
# curve_merge_thd : higher value -> curves merged more easily  lower_value -> curves are combined less ofter

CurvePiece = namedtuple("CurvePiece",['type','t_range','c_range'])

def findCurvePieces(path, line_detect_thd = 10, spiral_curve_slope_thd = 0.02, line_merge_thd = 2, curve_merge_thd=0.0000015):
	# first, identify all segments that are line-like or circle-like

	get_line_error_fn = lambda path, s0, s1: fitLine(*samplePosition(path, path.t2T(s0,0),path.t2T(s1-1,1)))[0]
	get_curve_error_fn = lambda path, s0, s1: fitLine(*sampleCurvature(path, path.t2T(s0,0),path.t2T(s1-1,1)))[0]

	classification = []

	# first, we classify each segment of the path as a line or as a curve
	for seg in path:
		if isLine(seg, 0.001, 0.999, line_detect_thd):
			classification.append("L") # line
		else:
			classification.append("C") # curve


	
	err_fn = {
		'L':(get_line_error_fn, line_merge_thd),
		'C':(get_curve_error_fn , curve_merge_thd),
		'c':(get_curve_error_fn , curve_merge_thd)
	}
	

	print("Total segments = ",len(path))

	print("Classification 1 = ",classification)
	# see which lines we can merge into a single line or turn to curves. 
	# No two consecutive lines allowed
	loop_again = True
	while loop_again:
		loop_again = False
		# try to merge more than one line segments into one. Whatever cannot be converted to single lines, gets turned into a curve
		for i,(t,rs) in enumerate(findRuns(classification)):
			a,b = rs
			if t=='L' and b > a+1: # runs of more than one line
				ranges = list(pickRanges(path, a, b, err_fn[t][0], err_fn[t][1]))
				if len(ranges) == 1: continue # ok, it got merged in a single line
				loop_again = True
				# pick the best fitting segment (longest better?), and turn the rest into curves
				print("Linear ranges:",ranges)
				ranges.sort(key = lambda rng: rng[1]) # sor according to error
				# turn the rest into curves
				for r,err in ranges[1:]: # keep the first one (best)
					for j in range(r[0],r[1]):
						classification[j+a] = 'c' # line turned to curve
			
	print("Classification 2 = ",classification)

	ranges = []
	for t,rs in findRuns(classification):
		print("Run: ",rs) # contiguous curves of the same type (ie: "CCC' or "LLLLLL" )
		# rs is such that classifications[rs[0]:rs[1]] gives all items in the run
		a,b = rs
		if b == a+1: # only one item in the run-> output it directly
			print("single")
			ranges.append((rs, t))
		else:
			# more than one item in the run -> merge them in disjoint ranges
			# not the range
			efn = err_fn[t]
			for r,err in pickRanges(path, a, b, efn[0], efn[1]):
				# results are indices into the array formed by a..b, so
				# we have to add a to have the actual value
				ra = r[0] + a, r[1] + a
				ranges.append((ra, t))
	
	print(ranges)
	ranges.sort()

	print("".join(r[1] for r in ranges))


	t_ranges = []
	for r,t in ranges:
		t01 = path.t2T(r[0],0),path.t2T(r[1]-1,1)
		if t == 'C': # check if it's a spiral or a circle
			err, m, b = fitLine(*sampleCurvature(path, *t01))
			if abs(m) > spiral_curve_slope_thd:
				t = 'S'
			c0 = m * t01[0] + b
			c1 = m * t01[1] + b
			t_ranges.append(CurvePiece(t, t01, (c0,c1)))
		else:
			t_ranges.append(CurvePiece(t, t01, (0,0))) 
			

	return t_ranges


def buildCurvesFromPieces(path, curve_pieces):
	tot_curves = len(curve_pieces)-1

	curves = []
	for curve_type, (t0,t1), c_range in curve_pieces:

		if curve_type == 'S':
			c = Spiral(path, t0, t1)
			print("spiral")
		elif curve_type == 'L':
			print("line")
			c = Line(path, t0, t1) 
		else:
			print("circle")
			c = Circle(path, t0, t1)

		if c:
			curves.append(c)

	prepareCurves(curves)

	return curves


				
def buildCurves(path, parameters, plot_curvature_axis=None, curves=None):
	# curve pieces" is a list of (curve_type, (from_t, to_t), (curv0, curv1))
	# that covers t from 0 to 1
	curve_pieces = findCurvePieces(path, **parameters['finding'])

	if plot_curvature_axis:
		for curve_type, t_range, c_range in curve_pieces:
			plot_curvature_axis.plot(t_range, c_range,'--')

	if not curves:
		curves = buildCurvesFromPieces(path, curve_pieces)

	return curves
