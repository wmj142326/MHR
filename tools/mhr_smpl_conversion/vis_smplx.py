# 导入配置和环境

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import json
import torch
import trimesh
import imageio
import pyrender
import numpy as np
import shutil
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel

# ---------- 构建 SMPL-X 输入 ----------
def to_t(x):
    """list/np -> torch (1, D) on device"""
    x = np.asarray(x, dtype=np.float32)
    t = torch.from_numpy(x).to(device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
body_model = BodyModel(bm_fname="./data/SMPLX_NEUTRAL.npz", num_betas=10, model_type="smplx").to(device).eval()

def look_at(eye, target, up=np.array([0, 1, 0], dtype=np.float32)):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    # OpenGL camera looks along -Z
    R = np.stack([right, true_up, -forward], axis=1)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = eye
    return pose

def build_fixed_camera_pose(center, radius, yfov=np.pi/3.0):
    """
    scene 内固定相机：只算一次 cam_pose,避免抖动
    """
    radius = max(float(radius), 1e-3)
    dist = radius / np.tan(yfov / 2.0) * 1.6
    eye = center + np.array([0.35 * radius, 0.15 * radius, dist], dtype=np.float32)
    cam_pose = look_at(eye, center)
    return cam_pose

def render_mesh_save_png_fixed_cam(
    mesh: trimesh.Trimesh,
    save_path: str,
    cam_pose: np.ndarray,
    W: int = 960,
    H: int = 960,
    yfov: float = np.pi / 3.0,
    normal_color: bool = True,
    renderer: pyrender.OffscreenRenderer | None = None,
):
    # 颜色
    if normal_color:
        colors = ((mesh.vertex_normals + 1.0) * 0.5 * 255).astype(np.uint8)
        mesh.visual.vertex_colors = colors
    else:
        mesh.visual.vertex_colors = np.tile(
            np.array([[180, 180, 180, 255]], dtype=np.uint8),
            (len(mesh.vertices), 1)
        )

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.35, 0.35, 0.35])

    mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_pr)

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    # 复用 renderer，别每帧创建（更快、更稳）
    local_renderer = renderer if renderer is not None else pyrender.OffscreenRenderer(W, H)
    try:
        color, depth = local_renderer.render(scene)
    finally:
        if renderer is None:
            local_renderer.delete()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, color)

def make_video_from_pngs(png_dir, out_mp4, fps=30):
    """
    将 png_dir 下按文件名排序的 png 合成 mp4
    """
    pngs = sorted([p for p in os.listdir(png_dir) if p.lower().endswith(".png")])
    if len(pngs) == 0:
        return False

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)
    try:
        for fn in pngs:
            img = imageio.imread(os.path.join(png_dir, fn))
            writer.append_data(img)
    finally:
        writer.close()
    return True

def main():
    pvcp_smplx_dir = "../../output/smplx/para"
    out_root = "/home/guest/wmj/Projects/MHR/output/mesh"
    W, H = 3500, 3500
    yfov = np.pi / 3.0
    fps = 30

    for scene_name in tqdm(sorted(os.listdir(pvcp_smplx_dir))):
        pvcp_smpl_file = os.path.join(pvcp_smplx_dir, scene_name)
        if not os.path.isfile(pvcp_smpl_file):
            continue

        with open(pvcp_smpl_file, "r") as f:
            data = json.load(f)

        # -------------------------
        # 1) 预扫一遍：统计该 scene 全局 bbox（用于固定相机）
        # -------------------------
        global_vmin = np.array([ np.inf,  np.inf,  np.inf], dtype=np.float32)
        global_vmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

        frame_keys = sorted(list(data.keys()))
        with torch.no_grad():
            for frame in frame_keys:
                pose_hand = np.concatenate([data[frame]["left_hand_pose"], data[frame]["right_hand_pose"]], axis=0)

                params = {
                    "root_orient": to_t(data[frame]["global_orient"]),
                    "trans":       to_t(data[frame]["transl"]),
                    "pose_body":   to_t(data[frame]["body_pose"]),
                    "betas":       to_t(data[frame]["betas"]),
                    "pose_hand":   to_t(pose_hand),
                    "expression":  to_t(data[frame]["expression"]),
                    "pose_jaw": torch.zeros(1, 3, device=device),
                    "pose_eye": torch.zeros(1, 6, device=device),
                }
                out = body_model(**params)
                verts = out.v[0].detach().cpu().numpy()
                global_vmin = np.minimum(global_vmin, verts.min(axis=0))
                global_vmax = np.maximum(global_vmax, verts.max(axis=0))

        center = (global_vmin + global_vmax) / 2.0
        radius = np.linalg.norm(global_vmax - global_vmin) / 2.0
        cam_pose = build_fixed_camera_pose(center, radius, yfov=yfov)

        # -------------------------
        # 2) 渲染该 scene 所有帧 png（相机固定）
        # -------------------------
        scene_png_dir = os.path.join(out_root, scene_name)
        os.makedirs(scene_png_dir, exist_ok=True)

        renderer = pyrender.OffscreenRenderer(W, H)
        try:
            with torch.no_grad():
                for frame in frame_keys:
                    pose_hand = np.concatenate([data[frame]["left_hand_pose"], data[frame]["right_hand_pose"]], axis=0)

                    params = {
                        "root_orient": to_t(data[frame]["global_orient"]),
                        "trans":       to_t(data[frame]["transl"]),
                        "pose_body":   to_t(data[frame]["body_pose"]),
                        "betas":       to_t(data[frame]["betas"]),
                        "pose_hand":   to_t(pose_hand),
                        "expression":  to_t(data[frame]["expression"]),
                        "pose_jaw": torch.zeros(1, 3, device=device),
                        "pose_eye": torch.zeros(1, 6, device=device),
                    }
                    out = body_model(**params)

                    verts = out.v[0].detach().cpu().numpy()
                    faces = body_model.f
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

                    # 确保文件名有 .png，且可排序（如果 frame 本身不是 png，就补）
                    fname = frame if str(frame).lower().endswith(".png") else f"{frame}.png"
                    save_path = os.path.join(scene_png_dir, fname)

                    render_mesh_save_png_fixed_cam(
                        mesh,
                        save_path,
                        cam_pose=cam_pose,
                        W=W, H=H, yfov=yfov,
                        normal_color=True,
                        renderer=renderer
                    )
        finally:
            renderer.delete()

        # -------------------------
        # 3) 合成视频到 out_root，并删除 scene 图片文件夹
        # -------------------------
        out_mp4 = os.path.join(out_root, f"{os.path.splitext(scene_name)[0]}.mp4")
        ok = make_video_from_pngs(scene_png_dir, out_mp4, fps=fps)

        if ok:
            shutil.rmtree(scene_png_dir, ignore_errors=True)
        else:
            print(f"[Warn] No pngs found for scene {scene_name}, keep folder: {scene_png_dir}")

if __name__ == "__main__":
    main()
