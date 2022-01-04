

# find spans or more than one equal elements in pieces
def findRuns(pieces):
	if not pieces:
		return

	p = None
	a = None
	b = None

	for b,q in enumerate(pieces):
		if p != q:
			if a is not None:
				yield p,(a,b)
			a,p = b,q
	yield p,(a,b+1)


# returns a list of pairs of indices of the largest spans in items that 
# it, and its subspans satisfy checker_fn
# Individual elements are returned only if they are not part of a larger segment
def getLargestSpans(tot_items, checker_fn, *args):
	# first step: find all the pairs that can be merged
	pairs = [(i,i) for i in range(tot_items)]
	# recursively merge all pairs
	while pairs:
		new_pairs = []
		possible_unlinked = 0
		unlinked = -1
		for i in range(len(pairs)-1):	
			can_link = checker_fn(pairs[i][0],pairs[i+1][1], *args)
			if can_link:
				if possible_unlinked == i: possible_unlinked = -1
				new_pairs.append((pairs[i][0],pairs[i+1][1]))
			else:
				unlinked = possible_unlinked
				possible_unlinked = i+1
			if unlinked >= 0:
				p = pairs[unlinked]
				yield p[0],p[1]+1
				unlinked = -1
		if possible_unlinked >= 0:
			p = pairs[possible_unlinked]
			yield p[0],p[1]+1
		pairs = new_pairs


# returns a list of pairs of indices of all the  spans in items that 
# it, and its subspans satisfy checker_fn
def getAllSpans(tot_items, checker_fn, *args):
	# first step: find all the pairs that can be merged
	pairs = [(i,i) for i in range(tot_items)]
	# recursively merge all pairs

	for i in range(tot_items):
		can_link = checker_fn(pairs[i][0],pairs[i][1], *args)
		if can_link:
			yield i, i+1

	while pairs:
		new_pairs = []
		for i in range(len(pairs)-1):	
			can_link = checker_fn(pairs[i][0],pairs[i+1][1], *args)
			if can_link:
				p =(pairs[i][0],pairs[i+1][1])
				new_pairs.append(p)
				yield p[0],p[1]+1
		pairs = new_pairs


class SpanTree:
	def __init__(self, spans):
		self._root = self._buildSpanTree(list(spans))

	def intersections(self, span):
		rs = set()

		def checkIntersection(node, span):
			if not node: return
			# check if touches either side
			if span[0] < node[3]: # node[3] = mid point
				checkIntersection(node[0], span) # check left side
			if span[1] > node[3]:
				checkIntersection(node[2], span) # check right side

			overlap = set()
			for s in node[1]: # check those contained in node
				if ((span[0] < s[1] and span[1] > s[0]) or 
						(s[0] < span[1] and s[1] > span[0])): 
					rs.add(s)

		checkIntersection(self._root, span)	

		return rs
		
	def add(self, span):
		node,prev_node,direc = self._find(span)
		if node: return

		if direc == 1:
			prev_node[2] = self._buildSpanTree([span])
		elif direc == -1:
			prev_node[0] = self._buildSpanTree([span])
		else:
			prev_node[1].append(span)
		 

	def remove(self, span):
		node,prev_node,direc = self._find(span)
		if not node: return False
		node[1].remove(span)
		return True


	def replace(self, span, *new_spans):
		node,prev_node,direc = self._find(span)
		if not node:
			return False

		node[1].remove(span)
		
		for s in new_spans:
			self.add(s)

		return True


	def iter(self):
		node_stack = [self._root]
		while node_stack:
			node = node_stack.pop()
			if node[0]: node_stack.append(node[0])
			if node[2]: node_stack.append(node[2])
			for span in node[1]:
				yield span
		


	def _find(self, span):
		node = self._root
		prev_node = None
		direction = 0

		while node:
			prev_node = node
			if span[0] > node[3]:
				node = node[2] # right
				direction = 1
			elif span[1] <= node[3]: # left
				node = node[0]
				direction = -1
			else:
				direction = 0
				for span_in_node in node[1]:
					if span_in_node:
						return node, prev_node, direction
				return None, node, direction
		return None, prev_node, direction
	

	def _buildSpanTree(self, spans):
		if not spans: return None

		# find a center element
		m_low = m_high = spans[0][0]
		for s in spans:
			m_low = min(m_low,min(s))	
			m_high = max(m_low,max(s))	
		mid_point = (m_high + m_low) // 2

		left = []
		center = []
		right = []

		for s in spans:
			if s[0]>mid_point:
				right.append(s)
			elif  s[1]<=mid_point:
				left.append(s)
			else:
				center.append(s)
				
		tree = [
			self._buildSpanTree(left),
			center,
			self._buildSpanTree(right),
			mid_point
			]

		return tree

	def print(self, node=None, indent = 0):
		if node is None:
			node = self._root
		ind = " "*indent
		print(f"{ind}Mid:{node[3]} Items:{node[1]}")
		print(f"{ind}Left:")
		if node[0] is None:
			print(f"{ind}  EMPTY")
		else:
			self.print(node[0], indent+2)
		print(f"{ind}Right:")
		if node[2] is None:
			print(f"{ind}  EMPTY")
		else:
			self.print(node[2], indent+2)
			

	def getOverlaps(self):
		intersecting = {}
		isolated = []
		for s in self.iter():
			inter = self.intersections(s)
			inter.remove(s)
			if not inter:
				isolated.append(s)
			else:
				intersecting[s] = inter

		return intersecting, isolated


	
def getDisjointSets(intersecting, isolated = None):
	all_nodes = set(intersecting.keys())
	disjoint_sets = {}

	gnum =-1 
	while all_nodes:
		gnum += 1
		n = all_nodes.pop()
		node_stack = [n]
		disjoint_sets[gnum] = []
		while node_stack:
			n = node_stack.pop()
			disjoint_sets[gnum].append(n)
			if n in intersecting:
				for connected in intersecting[n]:
					if connected in all_nodes:
						node_stack.append(connected)
						all_nodes.remove(connected)

	r = list(disjoint_sets.values())

	# these are already disjoint. Add them if provided
	if isolated:
		r += [[a] for a in isolated]

	return  r


# modifies the order of spans
def getOverlaps(spans):
	intersecting = {}
	isolated = []
	tree = SpanTree(spans)
	for s in spans:

	
		inter = tree.intersections(s)
		inter.remove(s)
		if not inter:
			isolated.append(s)
		else:
			intersecting[s] = inter

	return intersecting, isolated

		
			
def takeLowestIndex(items):
	best_i = 0
	best_fit = items[0]
	for i in range(1,len(items)):
		if best_fit > items[i]:
			best_fit = items[i]
			best_i   = i

	return best_i
	


if __name__ == "__main__":
	print(" RUNS -------------------------")

	runs = findRuns("AAABACCCCDCABBB")
	print(list(runs))

	checker = lambda lst, a, b: lst[b] - lst[a] < 5

	items = [0,1,2,3,4,5,6,8,9,10, 15, 20,23,25,26,27,28, 76,77]
	print(f"Item list: {items}")

	print(" SPANS -------------------------")

	# print between <> actual values of items (as opposed to indices into it)
	# note the values in <> include both extrema, unlike spans which don't include the second element
	def ss(s): return f"<{items[s[0]]},{items[s[1]-1]}>"

	spans = list(getLargestSpans(len(items), checker, items))
	print(f"Spans (indices) ---")

	for s in spans: print(ss(s))

	print(f"Span tree ---")

	st = SpanTree(spans)

	st.print()
	print("---")

	over, isolated = st.getOverlaps()
	print("Isolated")
	for ab in isolated: print(ss(ab))
	print("Overlapping")
	for k,v in over.items():
		print(f"{ss(k)} --->", ",".join(ss(ab) for ab in v))


	print(f"Disjoint spans ---")
	disjoint = getDisjointSets(over, isolated)
	for group in disjoint:
		print(",".join(ss(ab) for ab in group))

