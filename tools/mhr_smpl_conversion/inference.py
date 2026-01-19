# wmj-20260119
# MHR→SMPL(X)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import re
import json
import torch
import smplx
import numpy as np
from tqdm import tqdm
from mhr.mhr import MHR
from typing import Dict
from conversion import Conversion

from example import DEMO


def json_to_stacked_tensors(
    json_path: str,
    device: str = "cpu",
    sort_by_frame: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert a json file (frame -> {lbs, iden, expr}) into stacked torch tensors.

    Args:
        json_path: path to json file
        device: 'cpu', 'cuda', 'cuda:0', etc.
        sort_by_frame: whether to sort frames by name / index

    Returns:
        dict with torch.Tensor:
            {
              "lbs_model_params": (N, 204),
              "identity_coeffs":  (N, 45),
              "face_expr_coeffs": (N, 72),
            }
    """

    # -------- load json --------
    with open(json_path, "r", encoding="utf-8") as f:
        frame_dict = json.load(f)

    assert isinstance(frame_dict, dict), "JSON must be a dict: frame -> params"

    # -------- sort frames (optional) --------
    frames = list(frame_dict.keys())

    if sort_by_frame:
        # 优先按 frame 名字里的数字排序，如 frame_000123.png
        def extract_number(name):
            m = re.search(r"(\d+)", name)
            return int(m.group(1)) if m else -1

        frames = sorted(frames, key=extract_number)

    # -------- collect data --------
    lbs_list, iden_list, expr_list = [], [], []

    for frame in frames:
        v = frame_dict[frame]

        # 兼容不同字段命名
        lbs  = v.get("lbs_model_params", v.get("lbs"))
        iden = v.get("identity_coeffs",  v.get("iden"))
        expr = v.get("face_expr_coeffs", v.get("expr"))

        if lbs is None or iden is None or expr is None:
            raise KeyError(f"Missing keys in frame: {frame}")

        lbs  = np.asarray(lbs,  dtype=np.float32).squeeze()
        iden = np.asarray(iden, dtype=np.float32).squeeze()
        expr = np.asarray(expr, dtype=np.float32).squeeze()

        lbs_list.append(lbs)
        iden_list.append(iden)
        expr_list.append(expr)

    # -------- stack --------
    lbs_arr  = np.stack(lbs_list, axis=0)    # (N, 204)
    iden_arr = np.stack(iden_list, axis=0)   # (N, 45)
    expr_arr = np.stack(expr_list, axis=0)   # (N, 72)

    # -------- numpy -> torch --------
    tensor_data = {
        "lbs_model_params": torch.from_numpy(lbs_arr).to(device),
        "identity_coeffs":  torch.from_numpy(iden_arr).to(device),
        "face_expr_coeffs": torch.from_numpy(expr_arr).to(device),
    }

    return frames, tensor_data
 
 
# -------- helper: serialize one item (tensor/ndarray/list/number) --------
def to_jsonable(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x  # list / float / int / str / None ...


def get_smpl_model_file(model_type: str, base_dir: str = "./data"):
    """
    model_type: 'smpl' or 'smplx'
    return: path to model npz file
    """
    model_type = model_type.lower()

    if model_type == "smpl":
        return os.path.join(base_dir, "SMPL_NEUTRAL.npz")
    elif model_type == "smplx":
        return os.path.join(base_dir, "SMPLX_NEUTRAL.npz")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def extract_number(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def merge_smplx_para_folder(para_dir: str, out_path: str):
    """
    Merge per-scene SMPLX json files into one flat json.

    Input files:
        para_dir/S000_smplx.json
        para_dir/S001_smplx.json
        ...

    Each file format:
        { "frame_xxx.png": {param...}, ... }

    Output format:
        { "S000_frame_xxx.png": {param...}, ... }
    """

    files = [f for f in os.listdir(para_dir) if f.endswith(".json")]
    files = sorted(files, key=extract_number)

    merged = {}

    for fname in tqdm(files, desc="Merging SMPLX json"):
        scene_id = fname.split("_")[0]   # S000 from S000_smplx.json
        path = os.path.join(para_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{fname} is not frame->para dict.")

        frame_keys = sorted(list(data.keys()), key=extract_number)

        for frame_key in frame_keys:
            merged_key = f"{scene_id}_{frame_key}"   # S000_frame_000001.png

            if merged_key in merged:
                raise KeyError(f"Duplicate key: {merged_key}")

            merged[merged_key] = data[frame_key]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("Merged json saved to:", out_path)
    print("Total frames:", len(merged))


def main():
    demo=DEMO()
    device = demo._device
    print(device)
    model_type="smplx"
    model_file=get_smpl_model_file(model_type)
    scene_dir = "/home/guest/wmj/Projects/sam-3d-body/data/PVCP/sam3dbody_output/scene_json"

    # Initialize models
    mhr_model = MHR.from_files(lod=1, device=device)
    smplx_model = smplx.SMPLX(model_path=model_file, gender="neutral", use_pca=False, flat_hand_mean=True).to(device)


    # Create converter
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method="pytorch"
    )

    # Convert MHR back to SMPLX
    for scene_file in sorted(os.listdir(scene_dir)):
        print(f"----------{scene_file}----------")
        frame_keys, mhr_data = json_to_stacked_tensors(os.path.join(scene_dir,scene_file), device=device)
        
        smplx_results = converter.convert_mhr2smpl(
            mhr_parameters=mhr_data,
            single_identity=True,
            return_smpl_parameters=True,
            return_smpl_meshes=False,
        )

        # save smpl(x) json output
        para_dict = smplx_results.result_parameters  # dict with 7 keys
        N = len(frame_keys)
        sample_key = next(iter(para_dict.keys()))
        sample_val = para_dict[sample_key]

        if torch.is_tensor(sample_val) or isinstance(sample_val, np.ndarray) or isinstance(sample_val, list):
            # for list: assume length is N if per-frame list
            n0 = sample_val.shape[0] if hasattr(sample_val, "shape") else len(sample_val)
            assert n0 == N, f"Mismatch: frame_keys has N={N}, but para_dict['{sample_key}'] has first dim={n0}"

        # -------- build output: frame -> para --------
        out = {}
        for i, frame in enumerate(frame_keys):
            frame_para = {}
            for k, v in para_dict.items():
                # Per-frame tensor/array/list: take v[i]
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    frame_para[k] = to_jsonable(v[i])
                elif isinstance(v, list) and len(v) == N:
                    frame_para[k] = to_jsonable(v[i])
                else:
                    # global/shared metadata (not per-frame)
                    frame_para[k] = to_jsonable(v)
            out[frame] = frame_para

        para_output_dir = f'../../output/{model_type}/para'
        os.makedirs(para_output_dir, exist_ok=True)
        save_path = f"{para_output_dir}/{os.path.splitext(scene_file)[0]}_smplx.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved: {save_path}")


    smlpx_pvcp_json_path = os.path.dirname(para_output_dir)
    merge_smplx_para_folder(para_output_dir, f"{smlpx_pvcp_json_path}/pvcp_{model_type}.json")


if __name__ == '__main__':
    main()


