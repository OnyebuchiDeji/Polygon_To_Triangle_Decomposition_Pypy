"""
app.py
------

Polygon -> triangle decomposition, interactive.

All geometry lives in triangulate.py; this file owns input, state and drawing
only. Two independent rendering pipelines consume the same triangulation:

    PIPELINE 1 (diagonals) - dashed lines drawn from the diagonal index pairs,
                             i.e. the "conceptual" view the original app had.
    PIPELINE 2 (triangles) - filled/outlined triangles drawn from the triangle
                             index triples.

Controls
    left click    add a vertex
    right click   remove the vertex under the cursor
    1 / 2 / 3     diagonals / triangles / both
    L             toggle vertex + triangle index labels
    C             clear


On the Date: 05-09-2026

I added My Notes to understand how the code in `triangulate.py` works.
"""

from canvas import pg, Canvas
import triangulate as tri

vec2 = pg.math.Vector2

# Render pipeline flags (bitmask so "both" is just an OR).
MODE_DIAGONALS = 1
MODE_TRIANGLES = 2
MODE_BOTH = MODE_DIAGONALS | MODE_TRIANGLES
MODE_NAMES = {1: "diagonals", 2: "triangles", 3: "diagonals + triangles"}

COL_BG_TEXT = (235, 235, 235)
COL_POLY = (207, 109, 32)
COL_VERTEX = (32, 32, 196)
COL_DIAGONAL = (255, 255, 255)
COL_TRI_EDGE = (18, 18, 26)
COL_ERROR = (235, 96, 96)


class App:
	def __init__(self, title="", winWidth=1200, winHeight=675):
		self.canvas = Canvas(title, winWidth, winHeight, logics={
        	"update": self.update,
        	"draw":   self.draw,
    	})
		self.win_size = (winWidth, winHeight)
		self.font = self.canvas.get_font(size=15)
		self.hud_font = self.canvas.get_font(size=17)

		# --- polygon state -------------------------------------------------
		self.vertices = []          # list[vec2] - the authoritative point list
		self.sense_radius = 10

		# --- triangulation results (indices into self.vertices) ------------
		self.triangles = []         # list[(a, b, c)]
		self.diagonals = []         # list[(h, j)]
		self.status = "Add at least 3 vertices."
		self.status_ok = True

		# --- caching / input edge detection --------------------------------
		self.dirty = True           # recompute the triangulation next update
		self.prev_mouse = (False, False, False)
		self.prev_keys = set()

		# --- view options ---------------------------------------------------
		self.render_mode = MODE_BOTH
		self.show_labels = True
		self._fill_surface = None   # lazily created per-frame alpha surface

	# ------------------------------------------------------------------
	# Input
	# ------------------------------------------------------------------
	def sense_mouse(self):
		"""Edge-triggered mouse handling: one action per physical click."""
		pressed = pg.mouse.get_pressed()
		left = pressed[0] and not self.prev_mouse[0]
		right = pressed[2] and not self.prev_mouse[2]
		self.prev_mouse = pressed

		if not (left or right):
			return

		mouse_pos = vec2(pg.mouse.get_pos())

		if left:
			for vertex in self.vertices:
				if (mouse_pos - vertex).length() <= self.sense_radius * 2:
					return  # too close to an existing vertex; ignore
			self.vertices.append(mouse_pos)
			self.dirty = True

		elif right:
			for vertex in self.vertices:
				if (mouse_pos - vertex).length() <= self.sense_radius:
					self.vertices.remove(vertex)
					self.dirty = True
					break

	def sense_keys(self):
		keys = pg.key.get_pressed()
		now = {k for k in (pg.K_1, pg.K_2, pg.K_3, pg.K_l, pg.K_c) if keys[k]}
		fresh = now - self.prev_keys
		self.prev_keys = now

		if pg.K_1 in fresh:
			self.render_mode = MODE_DIAGONALS
		if pg.K_2 in fresh:
			self.render_mode = MODE_TRIANGLES
		if pg.K_3 in fresh:
			self.render_mode = MODE_BOTH
		if pg.K_l in fresh:
			self.show_labels = not self.show_labels
		if pg.K_c in fresh:
			self.vertices.clear()
			self.dirty = True

	# ------------------------------------------------------------------
	# Update
	# ------------------------------------------------------------------
	def update(self):
		self.sense_mouse()
		self.sense_keys()

		if not self.dirty:
			return
		self.dirty = False

		self.triangles = []
		self.diagonals = []

		if len(self.vertices) < 3:
			self.status = "Add at least 3 vertices."
			self.status_ok = True
			return

		result = tri.triangulate(self.vertices)
		self.triangles = result.triangles
		self.diagonals = result.diagonals
		self.status = result.message
		self.status_ok = result.ok

	# ------------------------------------------------------------------
	# Pipeline 1: dashed diagonals
	# ------------------------------------------------------------------
	def draw_diagonals(self, dash=6.0, gap=5.0, width=1):
		"""
		Walk each diagonal, alternating dash/gap. The final dash is clamped so
		the line always starts and ends on a dash without overshooting.
		"""
		for h, j in self.diagonals:
			p1 = self.vertices[h]
			p2 = self.vertices[j]
			delta = p2 - p1
			length = delta.length()
			if length < 1e-6:
				continue
			direction = delta / length

			travelled = 0.0
			while travelled < length:
				seg = min(dash, length - travelled)
				start = p1 + direction * travelled
				end = p1 + direction * (travelled + seg)
				pg.draw.line(self.canvas.window, COL_DIAGONAL, start, end, width)
				travelled += dash + gap

	# ------------------------------------------------------------------
	# Pipeline 2: filled triangles
	# ------------------------------------------------------------------
	@staticmethod
	def triangle_colour(index):
		"""Golden-angle hue stepping keeps adjacent triangles distinguishable."""
		colour = pg.Color(0, 0, 0)
		colour.hsva = ((index * 137.508) % 360.0, 62.0, 94.0, 100.0)
		return colour

	def draw_triangles(self, fill_alpha=120, outline_width=1):
		if not self.triangles:
			return

		if self._fill_surface is None:
			self._fill_surface = pg.Surface(self.win_size, pg.SRCALPHA)
		self._fill_surface.fill((0, 0, 0, 0))

		for n, (a, b, c) in enumerate(self.triangles):
			pts = (self.vertices[a], self.vertices[b], self.vertices[c])
			colour = self.triangle_colour(n)
			pg.draw.polygon(
				self._fill_surface,
				(colour.r, colour.g, colour.b, fill_alpha),
				pts,
			)

		self.canvas.window.blit(self._fill_surface, (0, 0))

		for n, (a, b, c) in enumerate(self.triangles):
			pts = (self.vertices[a], self.vertices[b], self.vertices[c])
			pg.draw.polygon(self.canvas.window, COL_TRI_EDGE, pts, outline_width)

			if self.show_labels:
				cx = (pts[0].x + pts[1].x + pts[2].x) / 3.0
				cy = (pts[0].y + pts[1].y + pts[2].y) / 3.0
				label = self.font.render(f"T{n}", True, COL_TRI_EDGE)
				self.canvas.window.blit(label, (cx - label.get_width() * 0.5,
										cy - label.get_height() * 0.5))

	# ------------------------------------------------------------------
	# Shared chrome
	# ------------------------------------------------------------------
	def draw_polygon_outline(self):
		if len(self.vertices) >= 3:
			pg.draw.polygon(self.canvas.window, COL_POLY, self.vertices, width=3)
		elif len(self.vertices) == 2:
			pg.draw.line(self.canvas.window, COL_POLY,
                         self.vertices[0], self.vertices[1], 3)

	def draw_vertices(self):
		for idx, vertex in enumerate(self.vertices):
			pg.draw.circle(self.canvas.window, COL_VERTEX, vertex, self.sense_radius)
			if self.show_labels:
				text = self.font.render(str(idx), True, (255, 255, 255))
				self.canvas.window.blit(
					text, (vertex.x - self.sense_radius, vertex.y + self.sense_radius))

	def draw_hud(self):
		lines = [
			f"mode [1/2/3]: {MODE_NAMES[self.render_mode]}",
			f"vertices: {len(self.vertices)}   "
			f"triangles: {len(self.triangles)}   diagonals: {len(self.diagonals)}",
			self.status,
		]
		colours = [COL_BG_TEXT, COL_BG_TEXT,
					COL_BG_TEXT if self.status_ok else COL_ERROR]
		for n, (line, colour) in enumerate(zip(lines, colours)):
			self.canvas.window.blit(self.hud_font.render(line, True, colour),
                                    (12, 10 + n * 20))

	def draw(self):
		if self.render_mode & MODE_TRIANGLES:
			self.draw_triangles()

		self.draw_polygon_outline()

		if self.render_mode & MODE_DIAGONALS:
			self.draw_diagonals()

		self.draw_vertices()
		self.draw_hud()

	def start(self):
		self.canvas.run()


if __name__ == "__main__":
	App("Polygon To Triangle Decomposition").start()
