#!/usr/bin/env python3

import sys
import os
import time
import h5py
import cv2
import numpy as np


def nearest_index(t_array, t):
    if t_array is None or len(t_array) == 0:
        return None

    idx = np.searchsorted(t_array, t)

    if idx <= 0:
        return 0

    if idx >= len(t_array):
        return len(t_array) - 1

    before = idx - 1
    after = idx

    if abs(t_array[before] - t) <= abs(t_array[after] - t):
        return before
    else:
        return after


def decode_jpeg(jpeg_array):
    arr = np.asarray(jpeg_array, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def make_info_panel(
    height,
    width,
    t,
    ft_ts,
    ft_wrench,
    pose_ts,
    pose_xyzrpy,
    gripper_ts,
    gripper_state,
    event_ts,
    event_labels,
    show_help=True,
):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    y = 32
    line = 24

    def put(text, x=20, color=(255, 255, 255), scale=0.56, thickness=1):
        nonlocal y
        cv2.putText(
            panel,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line

    put(f"Time: {t:.3f} sec", color=(0, 255, 255), scale=0.72, thickness=2)
    y += 8

    # FT
    put("[FT Wrench]", color=(255, 200, 0), scale=0.62, thickness=2)
    idx_ft = nearest_index(ft_ts, t)

    if idx_ft is not None and ft_wrench is not None and len(ft_wrench) > 0:
        w = ft_wrench[idx_ft]
        put(f"idx: {idx_ft} / {len(ft_ts)-1}")
        put(f"t_ft: {ft_ts[idx_ft]:.3f}")
        put(f"Fx: {w[0]: .3f} N")
        put(f"Fy: {w[1]: .3f} N")
        put(f"Fz: {w[2]: .3f} N")
        put(f"Tx: {w[3]: .4f} Nm")
        put(f"Ty: {w[4]: .4f} Nm")
        put(f"Tz: {w[5]: .4f} Nm")
    else:
        put("No FT data")

    y += 8

    # Pose
    put("[Pose xyzrpy]", color=(255, 200, 0), scale=0.62, thickness=2)
    idx_pose = nearest_index(pose_ts, t)

    if idx_pose is not None and pose_xyzrpy is not None and len(pose_xyzrpy) > 0:
        p = pose_xyzrpy[idx_pose]
        put(f"idx: {idx_pose} / {len(pose_ts)-1}")
        put(f"t_pose: {pose_ts[idx_pose]:.3f}")
        put(f"x:     {p[0]: .4f} m")
        put(f"y:     {p[1]: .4f} m")
        put(f"z:     {p[2]: .4f} m")
        put(f"roll:  {p[3]: .4f} rad")
        put(f"pitch: {p[4]: .4f} rad")
        put(f"yaw:   {p[5]: .4f} rad")
    else:
        put("No pose data")

    y += 8

    # Gripper
    put("[Gripper]", color=(255, 200, 0), scale=0.62, thickness=2)
    idx_grip = nearest_index(gripper_ts, t)

    if idx_grip is not None and gripper_state is not None and len(gripper_state) > 0:
        g = int(gripper_state[idx_grip])
        state_text = "open" if g == 0 else "close"

        put(f"idx: {idx_grip} / {len(gripper_ts)-1}")
        put(f"t_grip: {gripper_ts[idx_grip]:.3f}")
        put(
            f"state: {g} ({state_text})",
            color=(0, 255, 0) if g == 0 else (0, 0, 255),
            thickness=2,
        )
    else:
        put("No gripper data")

    y += 8

    # Event
    put("[Recent Event]", color=(255, 200, 0), scale=0.62, thickness=2)

    if event_ts is not None and len(event_ts) > 0:
        past_indices = np.where(event_ts <= t)[0]

        if len(past_indices) > 0:
            idx_evt = past_indices[-1]
            label = event_labels[idx_evt]

            if isinstance(label, bytes):
                label = label.decode("utf-8")

            put(f"t_event: {event_ts[idx_evt]:.3f}")
            put(f"label: {label}")
        else:
            put("No event yet")
    else:
        put("No events")

    # Help box: 오른쪽 아래 작게 표시
    if show_help:
        help_lines = [
            "space : play/pause",
            "a / d : prev / next",
            "v     : save mp4",
            "h     : hide help",
            "q/ESC : quit",
        ]

        scale = 0.43
        thickness = 1
        margin = 12
        line_h = 18
        box_w = 190
        box_h = line_h * len(help_lines) + 18

        x0 = width - box_w - margin
        y0 = height - box_h - margin

        overlay = panel.copy()
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + box_w, y0 + box_h),
            (35, 35, 35),
            -1,
        )
        panel = cv2.addWeighted(overlay, 0.75, panel, 0.25, 0)

        cv2.rectangle(
            panel,
            (x0, y0),
            (x0 + box_w, y0 + box_h),
            (100, 100, 100),
            1,
        )

        yy = y0 + 22

        for text in help_lines:
            cv2.putText(
                panel,
                text,
                (x0 + 10, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (190, 190, 190),
                thickness,
                cv2.LINE_AA,
            )
            yy += line_h

    return panel


def load_episode(path):
    f = h5py.File(path, "r")

    rgb_ts = f["rgb/timestamp"][:] if "rgb/timestamp" in f else np.array([])
    rgb_frames = f["rgb/frames"] if "rgb/frames" in f else None

    ft_ts = f["ft/timestamp"][:] if "ft/timestamp" in f else np.array([])
    ft_wrench = f["ft/wrench"][:] if "ft/wrench" in f else None

    pose_ts = f["pose/timestamp"][:] if "pose/timestamp" in f else np.array([])
    pose_xyzrpy = f["pose/xyzrpy"][:] if "pose/xyzrpy" in f else None

    gripper_ts = f["gripper/timestamp"][:] if "gripper/timestamp" in f else np.array([])
    gripper_state = f["gripper/state"][:] if "gripper/state" in f else None

    event_ts = f["events/timestamp"][:] if "events/timestamp" in f else np.array([])
    event_labels = f["events/label"][:] if "events/label" in f else []

    return {
        "file": f,
        "rgb_ts": rgb_ts,
        "rgb_frames": rgb_frames,
        "ft_ts": ft_ts,
        "ft_wrench": ft_wrench,
        "pose_ts": pose_ts,
        "pose_xyzrpy": pose_xyzrpy,
        "gripper_ts": gripper_ts,
        "gripper_state": gripper_state,
        "event_ts": event_ts,
        "event_labels": event_labels,
    }


def print_summary(data, path):
    f = data["file"]

    print("=" * 70)
    print("Episode file:", path)
    print("=" * 70)

    print("[attrs]")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

    print()
    print("[counts]")
    print("  RGB     :", len(data["rgb_ts"]))
    print("  FT      :", len(data["ft_ts"]))
    print("  Pose    :", len(data["pose_ts"]))
    print("  Gripper :", len(data["gripper_ts"]))
    print("  Events  :", len(data["event_ts"]))

    def approx_hz(ts):
        if ts is None or len(ts) <= 1:
            return 0.0

        duration = ts[-1] - ts[0]

        if duration <= 0:
            return 0.0

        return len(ts) / duration

    print()
    print("[approx Hz]")
    print(f"  RGB     : {approx_hz(data['rgb_ts']):.1f}")
    print(f"  FT      : {approx_hz(data['ft_ts']):.1f}")
    print(f"  Pose    : {approx_hz(data['pose_ts']):.1f}")
    print(f"  Gripper : {approx_hz(data['gripper_ts']):.1f}")

    print("=" * 70)


def render_frame(data, frame_idx, target_h=720, panel_w=520, show_help=True):
    rgb_ts = data["rgb_ts"]
    rgb_frames = data["rgb_frames"]

    img = decode_jpeg(rgb_frames[frame_idx])

    if img is None:
        return None

    # RGB 화면 왼쪽으로 90도 회전
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = img.shape[:2]
    scale = target_h / h
    img_resized = cv2.resize(img, (int(w * scale), target_h))

    t = float(rgb_ts[frame_idx])

    panel = make_info_panel(
        height=target_h,
        width=panel_w,
        t=t,
        ft_ts=data["ft_ts"],
        ft_wrench=data["ft_wrench"],
        pose_ts=data["pose_ts"],
        pose_xyzrpy=data["pose_xyzrpy"],
        gripper_ts=data["gripper_ts"],
        gripper_state=data["gripper_state"],
        event_ts=data["event_ts"],
        event_labels=data["event_labels"],
        show_help=show_help,
    )

    canvas = np.hstack([img_resized, panel])

    title = f"frame {frame_idx+1}/{len(rgb_ts)} | t={t:.3f}s"

    cv2.putText(
        canvas,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def save_viewer_video(data, input_path, show_help=True):
    rgb_ts = data["rgb_ts"]
    n = len(rgb_ts)

    if n == 0:
        print("No RGB frames. Cannot save video.")
        return

    if len(rgb_ts) > 1:
        duration = rgb_ts[-1] - rgb_ts[0]
        fps = len(rgb_ts) / duration if duration > 0 else 30.0
    else:
        fps = 30.0

    fps = max(1.0, min(60.0, float(fps)))

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.dirname(input_path)
    out_path = os.path.join(out_dir, f"{base}_viewer.mp4")

    first = render_frame(data, 0, show_help=show_help)

    if first is None:
        print("Failed to render first frame.")
        return

    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"Failed to open VideoWriter: {out_path}")
        return

    print("=" * 70)
    print(f"Saving viewer video: {out_path}")
    print(f"frames : {n}")
    print(f"fps    : {fps:.2f}")
    print(f"size   : {width} x {height}")
    print("=" * 70)

    for idx in range(n):
        frame = render_frame(data, idx, show_help=show_help)

        if frame is None:
            continue

        writer.write(frame)

        if idx % 50 == 0 or idx == n - 1:
            print(f"writing {idx+1}/{n}", end="\r")

    writer.release()

    print()
    print("=" * 70)
    print(f"Video saved: {out_path}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 view_episode_hdf5.py /path/to/episode_000000.h5")
        print("  python3 view_episode_hdf5.py /path/to/episode_000000.hdf5")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    data = load_episode(path)
    print_summary(data, path)

    rgb_ts = data["rgb_ts"]
    rgb_frames = data["rgb_frames"]

    if rgb_frames is None or len(rgb_ts) == 0:
        print("No RGB frames found. Cannot display video.")
        data["file"].close()
        sys.exit(1)

    playing = True
    show_help = True
    i = 0
    n = len(rgb_ts)

    window_name = "UMI-FT Episode Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_time = time.time()

    while True:
        canvas = render_frame(
            data,
            i,
            target_h=720,
            panel_w=520,
            show_help=show_help,
        )

        if canvas is None:
            print(f"Failed to render frame {i}")
            i += 1

            if i >= n:
                i = n - 1
                playing = False

            continue

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif key == ord(" "):
            playing = not playing

        elif key == ord("a"):
            playing = False
            i = max(0, i - 1)

        elif key == ord("d"):
            playing = False
            i = min(n - 1, i + 1)

        elif key == ord("h"):
            show_help = not show_help

        elif key == ord("v"):
            was_playing = playing
            playing = False
            save_viewer_video(data, path, show_help=show_help)
            playing = was_playing

        if playing:
            now = time.time()

            if i < n - 1:
                dt_data = rgb_ts[i + 1] - rgb_ts[i]
                dt_data = max(0.001, min(0.1, float(dt_data)))
            else:
                dt_data = 0.03

            elapsed = now - last_time

            if elapsed >= dt_data:
                i += 1
                last_time = now

                if i >= n:
                    i = n - 1
                    playing = False

    data["file"].close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()