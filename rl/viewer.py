import numpy as np
import pyray as rl


WINDOW_WIDTH = 792
WINDOW_HEIGHT = 872
HUD_HEIGHT = 80
PADDING = 12
BORDER = 3
MPS_TO_MPH = 2.23694
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
BACKGROUND = rl.Color(17, 19, 23, 255)
PANEL = rl.Color(27, 30, 36, 255)
FRAME = rl.Color(91, 101, 115, 255)
ACCENT = rl.Color(118, 169, 238, 255)
SUCCESS = rl.Color(109, 205, 151, 255)
INK = rl.Color(232, 235, 240, 255)


def _road_view(frame: np.ndarray) -> np.ndarray:
    """Stack the normal and wide RGB views into the model's familiar square."""
    if frame.ndim != 3 or frame.shape[-1] != 6:
        raise ValueError(f"expected an HxWx6 frame, got {frame.shape}")
    return np.ascontiguousarray(np.concatenate((frame[..., :3], frame[..., 3:]), axis=0))


class Viewer:
    """A window that knows about pixels, not environments."""

    def __init__(self, initial_frame: np.ndarray):
        frame = _road_view(initial_frame)
        self.height, self.width, _ = frame.shape
        rl.set_config_flags(rl.FLAG_WINDOW_RESIZABLE)
        rl.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Tiny world model rollout")
        rl.set_window_min_size(384, 464)
        rl.set_target_fps(60)

        image = rl.Image(frame, self.width, self.height, 1, rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
        self.texture = rl.load_texture_from_image(image)
        self.font = rl.load_font(FONT)
        rl.set_texture_filter(self.texture, rl.TEXTURE_FILTER_BILINEAR)
        rl.set_texture_filter(self.font.texture, rl.TEXTURE_FILTER_BILINEAR)

    @property
    def open(self) -> bool:
        return not rl.window_should_close()

    def render(
        self,
        frame: np.ndarray | None,
        speed: float,
        target: int,
        end: int,
        curvature: float,
        accel: float,
        complete: bool,
    ) -> None:
        if frame is not None:
            frame = _road_view(frame)
            rl.update_texture(self.texture, rl.ffi.cast("void *", frame.ctypes.data))

        screen_w, screen_h = rl.get_screen_width(), rl.get_screen_height()
        hud_y = screen_h - HUD_HEIGHT
        size = max(1, min(screen_w - 2 * PADDING, hud_y - 2 * PADDING))
        x, y = (screen_w - size) / 2, (hud_y - size) / 2
        destination = rl.Rectangle(x, y, size, size)
        status_text = "Complete  |  Esc to quit" if complete else "Running  |  Esc to quit"
        stats_size = 17 if screen_w >= 620 else 13
        speed *= MPS_TO_MPH
        stats = (
            f"Frame {target:04d} / {end:04d}    Speed {speed:.0f} mph    "
            f"Curvature {curvature:+.3f}    Accel {accel:+.1f}"
            if screen_w >= 620
            else f"{target:04d}/{end:04d}   {speed:.0f} mph   C {curvature:+.3f}   A {accel:+.1f}"
        )

        rl.begin_drawing()
        rl.clear_background(BACKGROUND)
        rl.draw_rectangle_rec(rl.Rectangle(x - BORDER, y - BORDER, size + 2 * BORDER, size + 2 * BORDER), FRAME)
        rl.draw_texture_pro(
            self.texture,
            rl.Rectangle(0, 0, self.width, self.height),
            destination,
            rl.Vector2(0, 0),
            0,
            rl.WHITE,
        )
        rl.draw_rectangle(0, hud_y, screen_w, HUD_HEIGHT, PANEL)
        rl.draw_rectangle(0, hud_y, screen_w, BORDER, FRAME)
        rl.draw_text_ex(self.font, "World model rollout", rl.Vector2(PADDING, hud_y + 9), 20, 0, INK)
        rl.draw_text_ex(
            self.font,
            status_text,
            rl.Vector2(screen_w - PADDING - rl.measure_text_ex(self.font, status_text, 16, 0).x, hud_y + 12),
            16,
            0,
            SUCCESS if complete else ACCENT,
        )
        rl.draw_text_ex(self.font, stats, rl.Vector2(PADDING, hud_y + 44), stats_size, 0, INK)
        rl.end_drawing()

    def close(self) -> None:
        rl.unload_texture(self.texture)
        rl.unload_font(self.font)
        rl.close_window()
