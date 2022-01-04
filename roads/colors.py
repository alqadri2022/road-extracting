import math

def hsv_to_rgb(h, s, v):
	if s == 0.0: return (v, v, v)
	i = int(h*6.) # XXX assume int() truncates!
	f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
	if i == 0: return (v, t, p)
	if i == 1: return (q, v, p)
	if i == 2: return (p, v, t)
	if i == 3: return (p, q, v)
	if i == 4: return (t, p, v)
	if i == 5: return (v, p, q)


def makeColorList(n, as_str = True):
	revs = 0
	colors = []
	golden = (1 + 5 ** 0.5) / 2
	for i in range(n):
		t = i / (n-1)
		revs +=  golden
		ang = math.fmod(revs,1)
		rgb = hsv_to_rgb(ang, math.cos(t*43)*0.2 + 0.6, math.sin(t*90)*0.2 + 0.6)
		if as_str:
			rgb = '#%02x%02x%02x' % (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))
		colors.append(rgb)
	return colors
	
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle
	def plot_colortable(colors):

		cell_width = 212
		cell_height = 22
		swatch_width = 48
		margin = 12
		topmargin = 40

		n = len(colors)
		ncols = 4
		nrows = n // ncols + int(n % ncols > 0)

		width = cell_width * 4 + 2 * margin
		height = cell_height * nrows + margin + topmargin
		dpi = 72

		fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
		fig.subplots_adjust(margin/width, margin/height,
							(width-margin)/width, (height-topmargin)/height)
		ax.set_xlim(0, cell_width * 4)
		ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
		ax.yaxis.set_visible(False)
		ax.xaxis.set_visible(False)
		ax.set_axis_off()
		for i, color in enumerate(colors):
			row = i % nrows
			col = i // nrows
			y = row * cell_height

			swatch_start_x = cell_width * col
			text_pos_x = cell_width * col + swatch_width + 7

			ax.add_patch(
				Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
						  height=18, facecolor=color, edgecolor='0.7')
			)

		return fig


	colors = makeColorList(100)

	plot_colortable(colors)
	plt.show()
