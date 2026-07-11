import cv2
import numpy as np
import os
import csv
import math
import tkinter as tk
from tkinter import filedialog, simpledialog


# ============================================================
# Configuration
# ============================================================

NUM_TICK_POINTS = 8
TICK_INTERVAL_MM = 5.0

WINDOW_NAME = "Ruler Perspective Size Estimation"

ZOOM_STEP = 1.25
MIN_ZOOM = 0.1
MAX_ZOOM = 20.0

PAN_STEP = 80

FONT_SCALE = 0.85
FONT_THICKNESS = 2
TOP_BAR_HEIGHT = 90


# ============================================================
# Unicode-safe image I/O
# ============================================================

def imread_unicode(path):
    """
    Read image from paths containing Chinese characters or network paths.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[Error] Failed to read image: {path}")
        print(e)
        return None


def imwrite_unicode(path, image):
    """
    Save image to paths containing Chinese characters or network paths.
    """
    try:
        ext = os.path.splitext(path)[1]
        if ext == "":
            ext = ".jpg"
            path = path + ext

        success, encoded_img = cv2.imencode(ext, image)
        if success:
            encoded_img.tofile(path)
            return True
        return False
    except Exception as e:
        print(f"[Error] Failed to save image: {path}")
        print(e)
        return False


# ============================================================
# File selection
# ============================================================

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return path


def select_label_folder():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="Select YOLO label folder")
    root.destroy()
    return path


def select_output_folder():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="Select output folder")
    root.destroy()
    return path


def ask_ruler_width_mm():
    """
    Optional:
    If the physical ruler width is known, it can be used for lateral scale.
    Otherwise, longitudinal local scale is used for both directions.
    """
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askstring(
        "Optional ruler width",
        "Input real ruler width in mm if known.\n"
        "Leave blank if unknown.\n\n"
        "Example: 15"
    )
    root.destroy()

    if value is None or value.strip() == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


# ============================================================
# YOLO label reading
# ============================================================

def read_yolo_labels(label_path, img_w, img_h):
    """
    YOLO format:
        class_id x_center y_center width height
    All coordinates are normalized.
    """
    objects = []

    if not os.path.exists(label_path):
        print(f"[Warning] Label file not found: {label_path}")
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = parts[0]

        try:
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue

        x_center = xc * img_w
        y_center = yc * img_h
        box_w = bw * img_w
        box_h = bh * img_h

        x1 = x_center - box_w / 2.0
        y1 = y_center - box_h / 2.0
        x2 = x_center + box_w / 2.0
        y2 = y_center + box_h / 2.0

        objects.append({
            "id": idx,
            "class_id": cls_id,
            "x_center": x_center,
            "y_center": y_center,
            "box_w_px": box_w,
            "box_h_px": box_h,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return objects


# ============================================================
# Geometry
# ============================================================

def fit_1d_projective_t_to_s(t_values, s_values):
    """
    Fit 1D projective transform:

        s(t) = (a * t + b) / (c * t + 1)

    Unknowns:
        a, b, c
    """
    t_values = np.asarray(t_values, dtype=np.float64)
    s_values = np.asarray(s_values, dtype=np.float64)

    A = []
    B = []

    for t, s in zip(t_values, s_values):
        A.append([t, 1.0, -s * t])
        B.append(s)

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = params

    return a, b, c


def projective_s(t, a, b, c):
    return (a * t + b) / (c * t + 1.0)


def local_ds_dt(t, a, b, c):
    """
    derivative of:
        s(t) = (a*t + b) / (c*t + 1)

    ds/dt = (a - b*c) / (c*t + 1)^2
    """
    denom = c * t + 1.0
    return (a - b * c) / (denom * denom)


def point_to_uv(point, origin, u_axis, v_axis):
    """
    Convert image coordinate to ruler local coordinate.

    u: along ruler direction
    v: perpendicular to ruler direction
    """
    p = np.asarray(point, dtype=np.float64)
    d = p - origin
    u = float(np.dot(d, u_axis))
    v = float(np.dot(d, v_axis))
    return u, v


def fit_edge_v_as_function_of_u(edge_points, origin, u_axis, v_axis):
    """
    Fit ruler edge in local coordinate:
        v = m*u + q
    """
    pts_uv = [point_to_uv(p, origin, u_axis, v_axis) for p in edge_points]

    us = np.array([p[0] for p in pts_uv], dtype=np.float64)
    vs = np.array([p[1] for p in pts_uv], dtype=np.float64)

    m = (vs[1] - vs[0]) / (us[1] - us[0] + 1e-9)
    q = vs[0] - m * us[0]

    return m, q


def ruler_width_pixels_at_u(u, top_line, bottom_line):
    m1, q1 = top_line
    m2, q2 = bottom_line

    v_top = m1 * u + q1
    v_bottom = m2 * u + q2

    return abs(v_top - v_bottom)


def evaluate_fit_error(tick_u_sorted, tick_s_mm, a, b, c):
    """
    Evaluate ruler tick fitting error in mm.
    """
    pred = np.array([projective_s(t, a, b, c) for t in tick_u_sorted])
    err = pred - tick_s_mm

    mean_abs_err = float(np.mean(np.abs(err)))
    max_abs_err = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    return pred, err, mean_abs_err, max_abs_err, rmse


# ============================================================
# Interactive annotation with zoom and pan
# ============================================================

class RulerAnnotator:
    def __init__(self, image):
        self.image = image.copy()
        self.img_h, self.img_w = image.shape[:2]

        self.stage = 0
        self.points_top = []
        self.points_bottom = []
        self.tick_points = []

        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.dragging = False
        self.last_mouse = None

        self.window_w = 1400
        self.window_h = 900

    def get_current_points(self):
        if self.stage == 0:
            return self.points_top
        elif self.stage == 1:
            return self.points_bottom
        elif self.stage == 2:
            return self.tick_points
        return []

    def reset_all(self):
        self.stage = 0
        self.points_top = []
        self.points_bottom = []
        self.tick_points = []
        self.update_display()

    def undo_last(self):
        pts = self.get_current_points()
        if len(pts) > 0:
            pts.pop()
        self.update_display()

    def confirm_stage(self):
        if self.stage == 0:
            if len(self.points_top) == 2:
                self.stage = 1
            else:
                print("[Info] Please click exactly TWO points on the upper edge.")
        elif self.stage == 1:
            if len(self.points_bottom) == 2:
                self.stage = 2
            else:
                print("[Info] Please click exactly TWO points on the lower edge.")
        elif self.stage == 2:
            if len(self.tick_points) == NUM_TICK_POINTS:
                self.stage = 3
            else:
                print(f"[Info] Please click exactly {NUM_TICK_POINTS} ruler tick points.")
        self.update_display()

    def instruction_text(self):
        if self.stage == 0:
            return f"Step 1/3: Click TWO points on the UPPER edge of the ruler. Current: {len(self.points_top)}/2"
        elif self.stage == 1:
            return f"Step 2/3: Click TWO points on the LOWER edge of the ruler. Current: {len(self.points_bottom)}/2"
        elif self.stage == 2:
            return f"Step 3/3: Click {NUM_TICK_POINTS} tick points. Adjacent ticks = {TICK_INTERVAL_MM:g} mm. Current: {len(self.tick_points)}/{NUM_TICK_POINTS}"
        else:
            return "Finished. Press Enter to continue."

    def help_text(self):
        return "Controls: mouse wheel/+/- zoom | right-drag or WASD pan | z undo | r reset | Enter confirm | q/Esc quit"

    def image_to_screen(self, p):
        x, y = p
        sx = int(round(x * self.zoom + self.offset_x))
        sy = int(round(y * self.zoom + self.offset_y + TOP_BAR_HEIGHT))
        return sx, sy

    def screen_to_image(self, x, y):
        ix = (x - self.offset_x) / self.zoom
        iy = (y - TOP_BAR_HEIGHT - self.offset_y) / self.zoom
        return ix, iy

    def zoom_at(self, factor, center_x=None, center_y=None):
        if center_x is None:
            center_x = self.window_w / 2
        if center_y is None:
            center_y = self.window_h / 2

        old_zoom = self.zoom
        new_zoom = np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM)

        if abs(new_zoom - old_zoom) < 1e-9:
            return

        # Keep the image point under cursor fixed
        img_x, img_y = self.screen_to_image(center_x, center_y)

        self.zoom = new_zoom

        self.offset_x = center_x - img_x * self.zoom
        self.offset_y = center_y - TOP_BAR_HEIGHT - img_y * self.zoom

        self.update_display()

    def pan(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self.update_display()

    def mouse_callback(self, event, x, y, flags, param):
        self.window_w = max(self.window_w, 1)
        self.window_h = max(self.window_h, 1)

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_at(ZOOM_STEP, x, y)
            else:
                self.zoom_at(1.0 / ZOOM_STEP, x, y)
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
            return

        if event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
            self.last_mouse = None
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.last_mouse is not None:
                dx = x - self.last_mouse[0]
                dy = y - self.last_mouse[1]
                self.offset_x += dx
                self.offset_y += dy
                self.last_mouse = (x, y)
                self.update_display()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.stage >= 3:
                return

            ix, iy = self.screen_to_image(x, y)

            if ix < 0 or iy < 0 or ix >= self.img_w or iy >= self.img_h:
                return

            p = (float(ix), float(iy))

            if self.stage == 0:
                if len(self.points_top) < 2:
                    self.points_top.append(p)

            elif self.stage == 1:
                if len(self.points_bottom) < 2:
                    self.points_bottom.append(p)

            elif self.stage == 2:
                if len(self.tick_points) < NUM_TICK_POINTS:
                    self.tick_points.append(p)

            self.update_display()

    def draw_point(self, canvas, p, color, label=None):
        sx, sy = self.image_to_screen(p)
        r = max(4, int(round(5 * min(self.zoom, 2.0))))
        cv2.circle(canvas, (sx, sy), r, color, -1)

        # Cross hair
        cv2.line(canvas, (sx - 10, sy), (sx + 10, sy), color, 1)
        cv2.line(canvas, (sx, sy - 10), (sx, sy + 10), color, 1)

        if label is not None:
            cv2.putText(
                canvas,
                str(label),
                (sx + 10, sy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2
            )

    def draw_line(self, canvas, p1, p2, color):
        s1 = self.image_to_screen(p1)
        s2 = self.image_to_screen(p2)
        cv2.line(canvas, s1, s2, color, 2)

    def update_display(self):
        # Resize image based on zoom
        disp_w = max(1, int(round(self.img_w * self.zoom)))
        disp_h = max(1, int(round(self.img_h * self.zoom)))

        resized = cv2.resize(self.image, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # Canvas size: current window approximate
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            if win_w > 100 and win_h > 100:
                self.window_w = win_w
                self.window_h = win_h
        except Exception:
            pass

        canvas = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        canvas[:, :] = (35, 35, 35)

        # Paste image to canvas
        x0 = int(round(self.offset_x))
        y0 = int(round(self.offset_y + TOP_BAR_HEIGHT))

        x1 = x0 + disp_w
        y1 = y0 + disp_h

        cx0 = max(0, x0)
        cy0 = max(TOP_BAR_HEIGHT, y0)
        cx1 = min(self.window_w, x1)
        cy1 = min(self.window_h, y1)

        if cx1 > cx0 and cy1 > cy0:
            rx0 = cx0 - x0
            ry0 = cy0 - y0
            rx1 = rx0 + (cx1 - cx0)
            ry1 = ry0 + (cy1 - cy0)
            canvas[cy0:cy1, cx0:cx1] = resized[ry0:ry1, rx0:rx1]

        # Top bar
        cv2.rectangle(canvas, (0, 0), (self.window_w, TOP_BAR_HEIGHT), (0, 0, 0), -1)

        cv2.putText(
            canvas,
            self.instruction_text(),
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS
        )

        cv2.putText(
            canvas,
            self.help_text(),
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (210, 210, 210),
            2
        )

        zoom_text = f"Zoom: {self.zoom:.2f}x"
        cv2.putText(
            canvas,
            zoom_text,
            (self.window_w - 190, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2
        )

        # Draw existing annotations
        for i, p in enumerate(self.points_top):
            self.draw_point(canvas, p, (0, 0, 255), f"T{i + 1}")
        if len(self.points_top) == 2:
            self.draw_line(canvas, self.points_top[0], self.points_top[1], (0, 0, 255))

        for i, p in enumerate(self.points_bottom):
            self.draw_point(canvas, p, (255, 0, 0), f"B{i + 1}")
        if len(self.points_bottom) == 2:
            self.draw_line(canvas, self.points_bottom[0], self.points_bottom[1], (255, 0, 0))

        for i, p in enumerate(self.tick_points):
            self.draw_point(canvas, p, (0, 255, 255), f"{i + 1}")

        cv2.imshow(WINDOW_NAME, canvas)

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.window_w, self.window_h)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        self.update_display()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                continue

            if key in [ord("q"), 27]:
                cv2.destroyAllWindows()
                return None

            elif key == ord("r"):
                self.reset_all()

            elif key == ord("z"):
                self.undo_last()

            elif key in [ord("+"), ord("=")]:
                self.zoom_at(ZOOM_STEP)

            elif key in [ord("-"), ord("_")]:
                self.zoom_at(1.0 / ZOOM_STEP)

            elif key == ord("w"):
                self.pan(0, PAN_STEP)

            elif key == ord("s"):
                self.pan(0, -PAN_STEP)

            elif key == ord("a"):
                self.pan(PAN_STEP, 0)

            elif key == ord("d"):
                self.pan(-PAN_STEP, 0)

            elif key in [13, 10]:  # Enter
                if self.stage < 3:
                    self.confirm_stage()
                else:
                    cv2.destroyAllWindows()
                    break

            if self.stage == 3:
                self.update_display()

        return {
            "top_edge": self.points_top,
            "bottom_edge": self.points_bottom,
            "tick_points": self.tick_points
        }


# ============================================================
# Estimation
# ============================================================

# def estimate_bbox_sizes(image, objects, annotation, ruler_width_mm=None):
#     top_edge = np.asarray(annotation["top_edge"], dtype=np.float64)
#     bottom_edge = np.asarray(annotation["bottom_edge"], dtype=np.float64)
#     tick_points = np.asarray(annotation["tick_points"], dtype=np.float64)
#
#     # Define ruler centerline
#     center_start = (top_edge[0] + bottom_edge[0]) / 2.0
#     center_end = (top_edge[1] + bottom_edge[1]) / 2.0
#
#     origin = center_start.copy()
#
#     u_axis = center_end - center_start
#     u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-9)
#
#     v_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float64)
#
#     # Project tick points to u coordinate
#     tick_u = []
#     for p in tick_points:
#         u, _ = point_to_uv(p, origin, u_axis, v_axis)
#         tick_u.append(u)
#
#     tick_u = np.asarray(tick_u, dtype=np.float64)
#
#     # Sort by ruler direction
#     order = np.argsort(tick_u)
#     tick_u_sorted = tick_u[order]
#
#     # Real distances: 0, 5, 10, ..., 35 mm
#     tick_s_mm = np.arange(len(tick_u_sorted), dtype=np.float64) * TICK_INTERVAL_MM
#
#     # Fit 1D projective mapping
#     a, b, c = fit_1d_projective_t_to_s(tick_u_sorted, tick_s_mm)
#
#     pred_s, err_s, mean_abs_err, max_abs_err, rmse = evaluate_fit_error(
#         tick_u_sorted,
#         tick_s_mm,
#         a,
#         b,
#         c
#     )
#
#     # Fit ruler upper/lower edges
#     top_line = fit_edge_v_as_function_of_u(top_edge, origin, u_axis, v_axis)
#     bottom_line = fit_edge_v_as_function_of_u(bottom_edge, origin, u_axis, v_axis)
#
#     results = []
#
#     for obj in objects:
#         cx = obj["x_center"]
#         cy = obj["y_center"]
#         bw = obj["box_w_px"]
#         bh = obj["box_h_px"]
#
#         center_u, center_v = point_to_uv((cx, cy), origin, u_axis, v_axis)
#
#         # Longitudinal local scale: mm/pixel
#         long_mm_per_px = abs(local_ds_dt(center_u, a, b, c))
#
#         # Avoid unreasonable numerical explosion
#         if not np.isfinite(long_mm_per_px) or long_mm_per_px <= 0:
#             long_mm_per_px = np.nan
#
#         # Lateral local scale
#         if ruler_width_mm is not None and np.isfinite(long_mm_per_px):
#             width_px_at_center = ruler_width_pixels_at_u(center_u, top_line, bottom_line)
#             if width_px_at_center > 1e-6:
#                 lateral_mm_per_px = ruler_width_mm / width_px_at_center
#             else:
#                 lateral_mm_per_px = long_mm_per_px
#         else:
#             lateral_mm_per_px = long_mm_per_px
#
#         # Estimate bbox diagonal in anisotropic local coordinate
#         diag_vec1 = np.array([bw, bh], dtype=np.float64)
#         diag_vec2 = np.array([bw, -bh], dtype=np.float64)
#
#         du1 = abs(np.dot(diag_vec1, u_axis))
#         dv1 = abs(np.dot(diag_vec1, v_axis))
#         du2 = abs(np.dot(diag_vec2, u_axis))
#         dv2 = abs(np.dot(diag_vec2, v_axis))
#
#         if np.isfinite(long_mm_per_px) and np.isfinite(lateral_mm_per_px):
#             diag1_mm = math.sqrt((du1 * long_mm_per_px) ** 2 + (dv1 * lateral_mm_per_px) ** 2)
#             diag2_mm = math.sqrt((du2 * long_mm_per_px) ** 2 + (dv2 * lateral_mm_per_px) ** 2)
#             diag_mean_mm = (diag1_mm + diag2_mm) / 2.0
#             diag_max_mm = max(diag1_mm, diag2_mm)
#         else:
#             diag1_mm = np.nan
#             diag2_mm = np.nan
#             diag_mean_mm = np.nan
#             diag_max_mm = np.nan
#
#         pixel_diag = math.sqrt(bw ** 2 + bh ** 2)
#
#         # Size class using 2 cm threshold
#         if np.isfinite(diag_mean_mm):
#             size_class_2cm = ">=2cm" if diag_mean_mm >= 20.0 else "<2cm"
#         else:
#             size_class_2cm = "unknown"
#
#         results.append({
#             "id": obj["id"],
#             "class_id": obj["class_id"],
#             "x_center_px": cx,
#             "y_center_px": cy,
#             "box_w_px": bw,
#             "box_h_px": bh,
#             "pixel_diag_px": pixel_diag,
#             "center_u_px": center_u,
#             "center_v_px": center_v,
#             "long_mm_per_px": long_mm_per_px,
#             "lateral_mm_per_px": lateral_mm_per_px,
#             "diag1_mm": diag1_mm,
#             "diag2_mm": diag2_mm,
#             "diag_mean_mm": diag_mean_mm,
#             "diag_max_mm": diag_max_mm,
#             "diag_mean_cm": diag_mean_mm / 10.0 if np.isfinite(diag_mean_mm) else np.nan,
#             "diag_max_cm": diag_max_mm / 10.0 if np.isfinite(diag_max_mm) else np.nan,
#             "size_class_2cm": size_class_2cm,
#             "x1": obj["x1"],
#             "y1": obj["y1"],
#             "x2": obj["x2"],
#             "y2": obj["y2"]
#         })
#
#     model_info = {
#         "origin": origin,
#         "u_axis": u_axis,
#         "v_axis": v_axis,
#         "projective_params": (a, b, c),
#         "tick_u_sorted": tick_u_sorted,
#         "tick_s_mm": tick_s_mm,
#         "pred_s_mm": pred_s,
#         "tick_error_mm": err_s,
#         "mean_abs_error_mm": mean_abs_err,
#         "max_abs_error_mm": max_abs_err,
#         "rmse_mm": rmse,
#         "top_line": top_line,
#         "bottom_line": bottom_line
#     }
#
#     return results, model_info

def estimate_bbox_sizes(image, objects, annotation, ruler_width_mm=None):
    top_edge = np.asarray(annotation["top_edge"], dtype=np.float64)
    bottom_edge = np.asarray(annotation["bottom_edge"], dtype=np.float64)
    tick_points = np.asarray(annotation["tick_points"], dtype=np.float64)

    # Define ruler centerline
    center_start = (top_edge[0] + bottom_edge[0]) / 2.0
    center_end = (top_edge[1] + bottom_edge[1]) / 2.0

    origin = center_start.copy()

    u_axis = center_end - center_start
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-9)

    v_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float64)

    # Project tick points to u coordinate
    tick_u = []
    for p in tick_points:
        u, _ = point_to_uv(p, origin, u_axis, v_axis)
        tick_u.append(u)

    tick_u = np.asarray(tick_u, dtype=np.float64)

    # Sort by ruler direction
    order = np.argsort(tick_u)
    tick_u_sorted = tick_u[order]

    # Real distances: 0, 5, 10, ..., 35 mm
    tick_s_mm = np.arange(len(tick_u_sorted), dtype=np.float64) * TICK_INTERVAL_MM

    # Fit 1D projective mapping only for checking ruler fitting quality
    a, b, c = fit_1d_projective_t_to_s(tick_u_sorted, tick_s_mm)

    pred_s, err_s, mean_abs_err, max_abs_err, rmse = evaluate_fit_error(
        tick_u_sorted,
        tick_s_mm,
        a,
        b,
        c
    )

    # ============================================================
    # Global single pixel resolution
    # ============================================================
    ruler_real_length_mm = (len(tick_u_sorted) - 1) * TICK_INTERVAL_MM
    ruler_pixel_length_px = abs(tick_u_sorted[-1] - tick_u_sorted[0])

    if ruler_pixel_length_px > 1e-6:
        global_mm_per_px = ruler_real_length_mm / ruler_pixel_length_px
    else:
        global_mm_per_px = np.nan

    results = []

    for obj in objects:
        cx = obj["x_center"]
        cy = obj["y_center"]
        bw = obj["box_w_px"]
        bh = obj["box_h_px"]

        center_u, center_v = point_to_uv((cx, cy), origin, u_axis, v_axis)

        pixel_diag = math.sqrt(bw ** 2 + bh ** 2)

        if np.isfinite(global_mm_per_px):
            diag_mean_mm = pixel_diag * global_mm_per_px
            diag_max_mm = diag_mean_mm
        else:
            diag_mean_mm = np.nan
            diag_max_mm = np.nan

        # Keep these column names for compatibility with the existing CSV writer
        long_mm_per_px = global_mm_per_px
        lateral_mm_per_px = global_mm_per_px
        diag1_mm = diag_mean_mm
        diag2_mm = diag_mean_mm

        # Size class using 2 cm threshold
        if np.isfinite(diag_mean_mm):
            size_class_2cm = ">=2cm" if diag_mean_mm >= 20.0 else "<2cm"
        else:
            size_class_2cm = "unknown"

        results.append({
            "id": obj["id"],
            "class_id": obj["class_id"],
            "x_center_px": cx,
            "y_center_px": cy,
            "box_w_px": bw,
            "box_h_px": bh,
            "pixel_diag_px": pixel_diag,
            "center_u_px": center_u,
            "center_v_px": center_v,
            "long_mm_per_px": long_mm_per_px,
            "lateral_mm_per_px": lateral_mm_per_px,
            "diag1_mm": diag1_mm,
            "diag2_mm": diag2_mm,
            "diag_mean_mm": diag_mean_mm,
            "diag_max_mm": diag_max_mm,
            "diag_mean_cm": diag_mean_mm / 10.0 if np.isfinite(diag_mean_mm) else np.nan,
            "diag_max_cm": diag_max_mm / 10.0 if np.isfinite(diag_max_mm) else np.nan,
            "size_class_2cm": size_class_2cm,
            "x1": obj["x1"],
            "y1": obj["y1"],
            "x2": obj["x2"],
            "y2": obj["y2"]
        })

    model_info = {
        "origin": origin,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "projective_params": (a, b, c),
        "tick_u_sorted": tick_u_sorted,
        "tick_s_mm": tick_s_mm,
        "pred_s_mm": pred_s,
        "tick_error_mm": err_s,
        "mean_abs_error_mm": mean_abs_err,
        "max_abs_error_mm": max_abs_err,
        "rmse_mm": rmse,
        "global_mm_per_px": global_mm_per_px,
        "ruler_real_length_mm": ruler_real_length_mm,
        "ruler_pixel_length_px": ruler_pixel_length_px
    }

    return results, model_info

# ============================================================
# Output
# ============================================================

def draw_results(image, results):
    vis = image.copy()

    for r in results:
        x1 = int(round(r["x1"]))
        y1 = int(round(r["y1"]))
        x2 = int(round(r["x2"]))
        y2 = int(round(r["y2"]))

        x1 = max(0, min(vis.shape[1] - 1, x1))
        x2 = max(0, min(vis.shape[1] - 1, x2))
        y1 = max(0, min(vis.shape[0] - 1, y1))
        y2 = max(0, min(vis.shape[0] - 1, y2))

        diag_cm = r["diag_mean_cm"]
        size_class = r["size_class_2cm"]

        if size_class == ">=2cm":
            color = (0, 0, 255)
        elif size_class == "<2cm":
            color = (0, 255, 0)
        else:
            color = (180, 180, 180)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if np.isfinite(diag_cm):
            text = f"{diag_cm:.2f} cm {size_class}"
        else:
            text = "unknown"

        cv2.putText(
            vis,
            text,
            (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

    return vis


def fmt2(x):
    """
    Keep two decimal places for numeric output.
    Return empty string for nan/inf.
    """
    try:
        if x is None or not np.isfinite(float(x)):
            return ""
        return f"{float(x):.2f}"
    except Exception:
        return x


def save_csv(results, csv_path):
    fieldnames = [
        "id",
        "class_id",
        "x_center_px",
        "y_center_px",
        "box_w_px",
        "box_h_px",
        "pixel_diag_px",
        "center_u_px",
        "center_v_px",
        "long_mm_per_px",
        "lateral_mm_per_px",
        "diag1_mm",
        "diag2_mm",
        "diag_mean_mm",
        "diag_max_mm",
        "diag_mean_cm",
        "diag_max_cm",
        "size_class_2cm"
    ]

    # 这些列保留 2 位小数
    numeric_2dec_cols = {
        "x_center_px",
        "y_center_px",
        "box_w_px",
        "box_h_px",
        "pixel_diag_px",
        "center_u_px",
        "center_v_px",
        "long_mm_per_px",
        "lateral_mm_per_px",
        "diag1_mm",
        "diag2_mm",
        "diag_mean_mm",
        "diag_max_mm",
        "diag_mean_cm",
        "diag_max_cm"
    }

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {}
            for k in fieldnames:
                if k in numeric_2dec_cols:
                    row[k] = fmt2(r[k])
                else:
                    row[k] = r[k]
            writer.writerow(row)


def save_model_info(model_info, path):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("Ruler perspective fitting information\n")
        f.write("=====================================\n\n")
        f.write(f"NUM_TICK_POINTS: {NUM_TICK_POINTS}\n")
        f.write(f"TICK_INTERVAL_MM: {TICK_INTERVAL_MM}\n\n")

        a, b, c = model_info["projective_params"]
        f.write("1D projective mapping:\n")
        f.write("s(t) = (a*t + b) / (c*t + 1)\n")
        f.write(f"a = {a}\n")
        f.write(f"b = {b}\n")
        f.write(f"c = {c}\n\n")

        f.write("Fitting error:\n")
        f.write(f"mean_abs_error_mm = {model_info['mean_abs_error_mm']:.6f}\n")
        f.write(f"max_abs_error_mm  = {model_info['max_abs_error_mm']:.6f}\n")
        f.write(f"rmse_mm           = {model_info['rmse_mm']:.6f}\n\n")

        f.write("Tick fitting details:\n")
        f.write("index,u_px,true_s_mm,pred_s_mm,error_mm\n")

        for i, (u, true_s, pred_s, err) in enumerate(zip(
            model_info["tick_u_sorted"],
            model_info["tick_s_mm"],
            model_info["pred_s_mm"],
            model_info["tick_error_mm"]
        )):
            f.write(f"{i},{u:.6f},{true_s:.6f},{pred_s:.6f},{err:.6f}\n")


# ============================================================
# Main
# ============================================================

def main():
    image_path = select_image_file()
    if not image_path:
        print("No image selected.")
        return

    label_folder = select_label_folder()
    if not label_folder:
        print("No label folder selected.")
        return

    output_folder = select_output_folder()
    if not output_folder:
        print("No output folder selected.")
        return

    image_path = os.path.normpath(image_path)
    label_folder = os.path.normpath(label_folder)
    output_folder = os.path.normpath(output_folder)

    ruler_width_mm = ask_ruler_width_mm()

    print("Image path:", image_path)
    print("Image exists:", os.path.exists(image_path))

    image = imread_unicode(image_path)

    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    img_h, img_w = image.shape[:2]

    image_name = os.path.basename(image_path)
    stem = os.path.splitext(image_name)[0]

    label_path = os.path.join(label_folder, stem + ".txt")

    print("Label path:", label_path)
    print("Label exists:", os.path.exists(label_path))

    objects = read_yolo_labels(label_path, img_w, img_h)

    print(f"Number of YOLO objects: {len(objects)}")

    if len(objects) == 0:
        print("[Warning] No objects found in label file.")

    annotator = RulerAnnotator(image)
    annotation = annotator.run()

    if annotation is None:
        print("Annotation cancelled.")
        return

    results, model_info = estimate_bbox_sizes(
        image=image,
        objects=objects,
        annotation=annotation,
        ruler_width_mm=ruler_width_mm
    )

    os.makedirs(output_folder, exist_ok=True)

    csv_path = os.path.join(output_folder, stem + "_size_estimation.csv")
    vis_path = os.path.join(output_folder, stem + "_size_estimation_vis.jpg")
    model_info_path = os.path.join(output_folder, stem + "_ruler_fit_info.txt")

    save_csv(results, csv_path)

    vis = draw_results(image, results)
    imwrite_unicode(vis_path, vis)

    save_model_info(model_info, model_info_path)

    print("\nFinished.")
    print(f"CSV saved to: {csv_path}")
    print(f"Visualization saved to: {vis_path}")
    print(f"Model fitting info saved to: {model_info_path}")

    print("\nRuler fitting error:")
    print(f"Mean absolute error: {model_info['mean_abs_error_mm']:.4f} mm")
    print(f"Max absolute error : {model_info['max_abs_error_mm']:.4f} mm")
    print(f"RMSE               : {model_info['rmse_mm']:.4f} mm")

    print("\nImportant note:")
    print("The estimated length is an approximate perspective-corrected projected size.")
    print("It is not a strict 3D measurement if the target and ruler are not coplanar.")


if __name__ == "__main__":
    main()