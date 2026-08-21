"""Landscape highlight -> 1080x1920 Shorts frame.

Reproduces, as one ffmpeg filtergraph, the DaVinci Resolve setup this replaced:
a landscape clip dropped on a portrait timeline, zoomed to 0.5, with a Blanking
Fill effect at 75% background opacity and no crop.

    +----------------+  <- blurred cover of the same frame, darkened to 75%
    |################|     (Blanking Fill)
    +----------------+
    |                |  <- the clip, scaled to canvas_height * zoom and
    |     source     |     centre-cropped to the canvas width
    |                |
    +----------------+
    |################|
    +----------------+

The one number worth being careful about is what zoom is measured against.
Resolve's zoom 1.0 here means FILL -- the clip scaled until it covers the
portrait frame, sides cropped off -- not fit, so the scale factor is
`canvas_height / source_height` and the foreground ends up `canvas_height *
zoom` tall whatever the source aspect. Blanking is therefore always top and
bottom, never left and right, which is why Blanking Fill only ever has bars to
fill. What varies with the source is how much WIDTH survives the crop: a
3440x1440 clip shows its central ~47% at zoom 0.5, a 1920x1080 one ~63%.

Both branches come off a single decode via `split`, so the source is read once.
Tunables live in config/shorts.json.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, video_io


def _even(value: float) -> int:
    """Round to an even integer -- yuv420p subsamples in 2x2 blocks."""
    return max(2, int(round(value / 2)) * 2)


@dataclass(frozen=True)
class Settings:
    """Everything config/shorts.json holds, after CLI overrides."""

    canvas_w: int
    canvas_h: int
    zoom: float
    offset_y: int
    opacity: float
    blur_sigma: float
    blur_downscale: int
    crop: tuple[int, int, int, int]  # left, right, top, bottom

    @classmethod
    def load(
        cls,
        *,
        zoom: float | None = None,
        offset_y: int | None = None,
        opacity: float | None = None,
        blur_sigma: float | None = None,
    ) -> "Settings":
        raw = config.load("shorts")
        canvas = raw["canvas"]
        background = raw["background"]
        crop = raw["crop"]
        return cls(
            canvas_w=int(canvas["width"]),
            canvas_h=int(canvas["height"]),
            zoom=float(raw["zoom"] if zoom is None else zoom),
            offset_y=int(raw["offset_y"] if offset_y is None else offset_y),
            opacity=float(
                background["opacity"] if opacity is None else opacity
            ),
            blur_sigma=float(
                background["blur_sigma"] if blur_sigma is None else blur_sigma
            ),
            blur_downscale=int(background["blur_downscale"]),
            crop=(
                int(crop["left"]),
                int(crop["right"]),
                int(crop["top"]),
                int(crop["bottom"]),
            ),
        )


@dataclass(frozen=True)
class Layout:
    """The composite's geometry, resolved against one source resolution."""

    source_w: int
    source_h: int
    canvas_w: int
    canvas_h: int
    zoom: float
    offset_y: int
    scale_w: int  # foreground after scaling, before any cropping
    scale_h: int
    visible_w: int  # what survives the crop to the canvas
    visible_h: int
    crop: tuple[int, int, int, int]

    @property
    def bar_h(self) -> int:
        """Blanking above and below the foreground -- what the fill covers."""
        return (self.canvas_h - self.visible_h) // 2

    @property
    def bar_w(self) -> int:
        """Blanking left and right. 0 for any landscape source at zoom <= 1."""
        return (self.canvas_w - self.visible_w) // 2

    @property
    def width_kept(self) -> float:
        """Fraction of the source's width still on screen."""
        return self.visible_w / self.scale_w if self.scale_w else 0.0

    def describe(self) -> str:
        lines = [
            f"  source     {self.source_w}x{self.source_h}",
            f"  canvas     {self.canvas_w}x{self.canvas_h}",
            f"  zoom {self.zoom:g}     scaled to {self.scale_w}x{self.scale_h}"
            f", showing {self.visible_w}x{self.visible_h}"
            f" ({self.width_kept:.0%} of the width)",
        ]
        if self.bar_h > 0:
            lines.append(f"  blanking   {self.bar_h}px top and bottom, blur-filled")
        else:
            lines.append("  blanking   none at this zoom -- the fill never shows")
        if self.bar_w > 0:
            lines.append(f"             {self.bar_w}px left and right")
        if self.offset_y:
            direction = "down" if self.offset_y > 0 else "up"
            lines.append(f"  offset     {abs(self.offset_y)}px {direction} of centre")
        return "\n".join(lines)


def scaled_size(
    source_w: int, source_h: int, canvas_h: int, zoom: float
) -> tuple[int, int]:
    """The size the source is scaled to at `zoom`, before any cropping.

    Zoom 1.0 == fill the canvas height, which is what Resolve's baseline scale
    does for a landscape clip on a portrait timeline.
    """
    scale_h = _even(canvas_h * zoom)
    return _even(source_w * scale_h / source_h), scale_h


def zoom_for_height(scale_h: float, canvas_h: int) -> float:
    """The zoom that scales the source to `scale_h` pixels tall."""
    return scale_h / canvas_h


def zoom_for_width(
    scale_w: float, source_w: int, source_h: int, canvas_h: int
) -> float:
    """The zoom that scales the source to `scale_w` pixels wide.

    The aspect is the source's, so a width is just a height said differently.
    """
    return scale_w * source_h / source_w / canvas_h


def plan(info: video_io.VideoInfo, settings: Settings) -> Layout:
    """Resolve the composite's geometry for one source video."""
    if info.width <= 0 or info.height <= 0:
        raise ValueError(f"Unusable source resolution: {info.width}x{info.height}")
    if settings.zoom <= 0:
        raise ValueError(f"Zoom must be positive, got {settings.zoom}")

    scale_w, scale_h = scaled_size(
        info.width, info.height, settings.canvas_h, settings.zoom
    )

    left, right, top, bottom = settings.crop
    inset_w = max(2, scale_w - left - right)
    inset_h = max(2, scale_h - top - bottom)

    return Layout(
        source_w=info.width,
        source_h=info.height,
        canvas_w=settings.canvas_w,
        canvas_h=settings.canvas_h,
        zoom=settings.zoom,
        offset_y=settings.offset_y,
        scale_w=scale_w,
        scale_h=scale_h,
        visible_w=min(settings.canvas_w, inset_w),
        visible_h=min(settings.canvas_h, inset_h),
        crop=settings.crop,
    )


def filtergraph(layout: Layout, settings: Settings) -> tuple[str, str]:
    """Build the filter_complex for `layout`. Returns (graph, output label).

    One decode feeds both branches through `split`. The background is the same
    frame scaled to cover the canvas, blurred and darkened; the foreground is
    laid over it.
    """
    canvas_w, canvas_h = layout.canvas_w, layout.canvas_h

    # Blurring a full-size 1080x1920 plane at the radius this needs is by far
    # the most expensive thing in the graph, and pointless: shrink, blur small,
    # scale back up. Effective sigma is blur_sigma * blur_downscale.
    small_w = _even(canvas_w / settings.blur_downscale)
    small_h = _even(canvas_h / settings.blur_downscale)
    opacity = max(0.0, min(1.0, settings.opacity))

    background = (
        f"[bg]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
        f"crop={canvas_w}:{canvas_h},"
        f"scale={small_w}:{small_h},"
        f"gblur=sigma={settings.blur_sigma:g},"
        f"scale={canvas_w}:{canvas_h}:flags=bicubic,"
        # A straight multiply toward black -- what Blanking Fill's background
        # opacity comes to over the timeline's black backdrop.
        f"colorlevels=romax={opacity:g}:gomax={opacity:g}:bomax={opacity:g}"
        f"[backdrop]"
    )

    left, right, top, bottom = layout.crop
    steps = [f"scale={layout.scale_w}:{layout.scale_h}"]
    if any(layout.crop):
        steps.append(
            f"crop={max(2, layout.scale_w - left - right)}:"
            f"{max(2, layout.scale_h - top - bottom)}:{left}:{top}"
        )
    # Centre-crop to the canvas. min() keeps this a no-op for a source narrower
    # than the canvas (or a zoom under 1) instead of failing; commas inside a
    # filter argument have to be escaped from the graph's own separator.
    steps.append(rf"crop=min({canvas_w}\,iw):min({canvas_h}\,ih)")
    foreground = f"[fg]{','.join(steps)}[front]"

    sign = "+" if layout.offset_y >= 0 else "-"
    overlay = (
        f"[backdrop][front]overlay=(W-w)/2:(H-h)/2{sign}{abs(layout.offset_y)}"
        f",setsar=1[out]"
    )

    graph = ";".join(["[0:v]split=2[bg][fg]", background, foreground, overlay])
    return graph, "out"
