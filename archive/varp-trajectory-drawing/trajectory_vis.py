"""
Shared trajectory visualisation utilities.

Used by both human collection (collect_trajectories_varp.py) and
simulated collection pipelines.
"""

import cv2
import matplotlib.cm as cm
import numpy as np


def project_eef_to_image(eef_positions, camera_transform, img_height, img_width):
    """
    Project a list of 3-D world-space EEF positions to 2-D pixel coordinates.

    Args:
        eef_positions (np.ndarray): (T, 3) world-frame EEF positions
        camera_transform (np.ndarray): 4x4 world-to-pixel matrix from
            get_camera_transform_matrix(), captured while sim is still live.
        img_height (int): image height in pixels
        img_width (int): image width in pixels

    Returns:
        pixels (list[tuple|None]): (u, v) per timestep, or None if outside frame
    """
    pixels = []
    for pos in eef_positions:
        pt_h = np.array([pos[0], pos[1], pos[2], 1.0])
        proj = camera_transform @ pt_h   # → [u*d, v*d, d, 1]
        depth = proj[2]
        if depth <= 0:
            pixels.append(None)
            continue
        u = int(proj[0] / depth)
        v = int(proj[1] / depth)
        if 0 <= u < img_width and 0 <= v < img_height:
            pixels.append((u, v))
        else:
            pixels.append(None)
    return pixels


def capture_clean_frame(env, camera_name, img_height, img_width):
    """
    Render one camera frame with all VisualizationWrapper indicators hidden.

    Finds the VisualizationWrapper in the env chain, temporarily disables all
    visualizations (zeroing their site RGBAs), renders directly via
    env.sim.render(), then restores everything.

    Returns:
        frame (np.ndarray): (H, W, 3) uint8 RGB image in OpenGL convention
            (row 0 = bottom). Caller should flip vertically before saving.
    """
    viz_wrapper = None
    node = env
    while node is not None:
        if hasattr(node, "_vis_settings"):
            viz_wrapper = node
            break
        node = getattr(node, "env", None)

    if viz_wrapper is not None:
        orig_settings = dict(viz_wrapper._vis_settings)
        viz_wrapper.env.visualize(vis_settings={k: False for k in orig_settings})

    frame = env.sim.render(width=img_width, height=img_height, camera_name=camera_name)

    if viz_wrapper is not None:
        viz_wrapper.env.visualize(vis_settings=orig_settings)

    return frame  # (H, W, 3) uint8 RGB, OpenGL convention


def draw_trajectory_on_image(image, pixels, colormap="plasma", dot_radius=4, line_thickness=2):
    """
    Draw a temporally color-coded trajectory overlay on an image.

    Early timesteps are drawn in the cool end of the colormap; late timesteps
    in the warm end (matching the VARP paper's visualisation convention).

    Args:
        image (np.ndarray): (H, W, 3) uint8 BGR image (OpenCV format)
        pixels (list[tuple|None]): projected (u, v) coordinates per timestep
        colormap (str): matplotlib colormap name
        dot_radius (int): radius of each trajectory dot
        line_thickness (int): thickness of connecting lines

    Returns:
        vis (np.ndarray): annotated image copy
    """
    vis = image.copy()
    cmap = cm.get_cmap(colormap)

    valid = [(i, p) for i, p in enumerate(pixels) if p is not None]
    n = len(valid)
    if n == 0:
        return vis

    for idx, (i, (u, v)) in enumerate(valid):
        t = idx / max(n - 1, 1)
        r, g, b, _ = cmap(t)
        color_bgr = (int(b * 255), int(g * 255), int(r * 255))

        if idx > 0:
            _, (pu, pv) = valid[idx - 1]
            cv2.line(vis, (pu, pv), (u, v), color_bgr, line_thickness, cv2.LINE_AA)

        cv2.circle(vis, (u, v), dot_radius, color_bgr, -1, cv2.LINE_AA)

    _, (su, sv) = valid[0]
    _, (eu, ev) = valid[-1]
    cv2.circle(vis, (su, sv), dot_radius + 3, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(vis, (eu, ev), (255, 255, 255), cv2.MARKER_CROSS, dot_radius * 3, 2, cv2.LINE_AA)

    return vis


def add_colorbar_legend(vis_image, colormap="plasma", label="Time"):
    """Append a vertical colorbar legend on the right side of the image."""
    H = vis_image.shape[0]
    bar_w = 30
    padding = 10
    bar_h = H - 2 * padding

    gradient = np.linspace(1, 0, bar_h)[:, None]
    cmap = cm.get_cmap(colormap)
    bar_rgb = (cmap(gradient)[:, :, :3] * 255).astype(np.uint8)
    bar_bgr = bar_rgb[:, :, ::-1]
    bar_bgr = np.repeat(bar_bgr, bar_w, axis=1)

    legend = np.zeros((H, bar_w + 50, 3), dtype=np.uint8)
    legend[padding:padding + bar_h, :bar_w] = bar_bgr

    cv2.putText(legend, label, (2, padding - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(legend, "end", (bar_w + 3, padding + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(legend, "start", (bar_w + 3, padding + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return np.hstack([vis_image, legend])
