import sys
import os
import pygame as pg
from PIL import Image
from tkinter.simpledialog import askstring

class Canvas:
	def __init__(
		self, title="",
		width=1200, height=675,
		logics=None):

		pg.init()
		self.win_size = (width, height)
		self.window = pg.display.set_mode(self.win_size)
		self.title = title
		self.clock = pg.time.Clock()
		self.fps = 60
		self.time_elapsed = None
		self.prev_time = None
		self.delta_time = None
		self.framerate = None
		self.clear_color = (0, 0, 0)

		self.logics: dict[str, None] = logics
		if not "update" in self.logics or not "draw" in self.logics:
			raise Exception("Logics must include update and draw key-function object pairs")

	def update_caption(self):
		pg.display.set_caption(f"{self.title.title()} | fps: {self.clock.get_fps(): .4f}")

	def set_clear_color(self, col: tuple[int,int,int]):
		self.clear_color = col


	def get_font(self, style: str = "Verdana", size: int = 60):
		return pg.font.SysFont(style, size)

	def on_exit(self):
		if "at_exit" in self.logics:
			self.logics['at_exit']()
		pg.quit()
		sys.exit()
		exit()

	def poll_events(self):
		for e in pg.event.get():
			if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
				self.on_exit()
			elif e.type == pg.KEYDOWN and e.key == pg.K_s:
				self.save_pixels_as_image()

	def update(self):
		self.time_elapsed = pg.time.get_ticks() * 0.001
		self.delta_time = self.time_elapsed - self.prev_time if self.prev_time else 0
		self.prev_time = self.time_elapsed
		self.update_caption()
		self.logics['update']()

	def draw(self):
		self.window.fill(self.clear_color)
		self.logics['draw']()
		pg.display.flip()

	def run(self):
		while True:
			self.poll_events()
			self.update()
			self.draw()
			self.framerate = self.clock.tick(self.fps)

	def save_pixels_as_image(self, imageFormat: str="png"):
		image_name = askstring("Image Name", "Save Image As", initialvalue="_")
		if not image_name:
			print("Cancelled. Image Not Saved.")
			return

		save_dir = os.path.join(os.path.dirname(__file__), "..", "_scrnshots", f"{image_name}.{imageFormat}")
		scrn_size = self.window.get_size()
		if imageFormat == "png":
			raw_px_data = pg.image.tobytes(self.window, "RGBA")
			image = Image.frombytes("RGBA", scrn_size, raw_px_data)
		else:
			raw_px_data = pg.image.tobytes(self.window, "RGB")
			image = Image.frombytes("RGB", scrn_size, raw_px_data)

		image.save(save_dir)
		print(f"Saved Image {image_name} in directory, {save_dir}")