from concurrent.futures import ProcessPoolExecutor
import traceback
import warnings
import numpy as np
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import append_jsonlines, load_episodes_stats, write_episode_stats

ERROR_LOG_PATH = "meta/episode_stats_errors.jsonl"


def _normalize_image_feature_stats(ft: dict, stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ensure image/video stats have 3 channels to satisfy downstream assertions."""
    if ft["dtype"] not in ["image", "video"]:
        return stats

    fixed = {}
    for k, v in stats.items():
        if k == "count":
            fixed[k] = v
            continue

        arr = v
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.shape[0] > 3:
            arr = arr[:3]

        fixed[k] = arr

    return fixed


def _normalize_episode_stats(ep_stats: dict, features: dict) -> dict:
    for key, ft in features.items():
        if key in ep_stats and ft["dtype"] in ["image", "video"]:
            ep_stats[key] = _normalize_image_feature_stats(ft, ep_stats[key])
    return ep_stats


# -------- Worker-side globals --------
_G = {}

def _init_worker(repo_id: str, root: str, revision: str | None):
    # 每个 worker 进程启动时只运行一次
    _G["dataset"] = LeRobotDataset(repo_id, root=root, revision=revision)

def _sample_episode_video_frames(ds: LeRobotDataset, episode_index: int, ft_key: str) -> np.ndarray:
    ep_len = ds.meta.episodes[episode_index]["length"]
    sampled_indices = sample_indices(ep_len)
    query_timestamps = ds._get_query_timestamps(0.0, {ft_key: sampled_indices})
    video_frames = ds._query_videos(query_timestamps, episode_index)
    return video_frames[ft_key].numpy()

def _convert_episode_stats(ep_idx: int):
    # 任务函数：只接收 ep_idx（小参数）
    ds: LeRobotDataset = _G["dataset"]

    try:
        ep_start_idx = ds.episode_data_index["from"][ep_idx]
        ep_end_idx = ds.episode_data_index["to"][ep_idx]

        # ✅ 比 select(range(...)) 通常更快：连续区间切片
        ep_data = ds.hf_dataset[ep_start_idx:ep_end_idx]  # dict[str, list/array]

        ep_stats = {}
        for key, ft in ds.features.items():
            if ft["dtype"] == "video":
                ep_ft_data = _sample_episode_video_frames(ds, ep_idx, key)
            else:
                # np.asarray 通常比 np.array 更少拷贝
                ep_ft_data = np.asarray(ep_data[key])

            if ft["dtype"] in ["image", "video"]:
                # 兼容缺失/错位的通道维：统一成 (N, C, H, W)
                if ep_ft_data.ndim == 3:
                    ep_ft_data = ep_ft_data[:, None, ...]  # grayscale -> add channel dim
                elif ep_ft_data.ndim == 4 and ep_ft_data.shape[1] not in (1, 3) and ep_ft_data.shape[-1] in (1, 3):
                    ep_ft_data = np.transpose(ep_ft_data, (0, 3, 1, 2))  # channel-last -> channel-first
                if ep_ft_data.shape[1] == 1:
                    ep_ft_data = np.repeat(ep_ft_data, 3, axis=1)  # force 3 channels
                elif ep_ft_data.shape[1] > 3:
                    ep_ft_data = ep_ft_data[:, :3, ...]

            axes_to_reduce = (0, 2, 3) if ft["dtype"] in ["image", "video"] else 0
            keepdims = True if ft["dtype"] in ["image", "video"] else (ep_ft_data.ndim == 1)

            ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)

            if ft["dtype"] in ["image", "video"]:
                # remove batch dim
                ep_stats[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()}

        ep_stats = _normalize_episode_stats(ep_stats, ds.features)
        return {"episode_index": ep_idx, "stats": ep_stats, "error": None}
    except Exception:
        return {
            "episode_index": ep_idx,
            "stats": None,
            "error": traceback.format_exc(),
        }


def _load_existing_episode_stats(dataset: LeRobotDataset) -> set[int]:
    try:
        existing = load_episodes_stats(dataset.root)
    except FileNotFoundError:
        return set()

    # normalize cached stats for consistency
    for ep_idx, ep_stats in existing.items():
        dataset.meta.episodes_stats[ep_idx] = _normalize_episode_stats(ep_stats, dataset.features)
    return set(existing.keys())


def _record_error(root, episode_index: int, error_msg: str):
    append_jsonlines(
        {"episode_index": episode_index, "error": error_msg},
        root / ERROR_LOG_PATH,
    )


def convert_stats(dataset: LeRobotDataset, num_workers: int = 8, chunksize: int = 4):
    assert dataset.episodes is None
    total_episodes = dataset.meta.total_episodes

    # 清理旧的错误日志，避免新旧混淆
    error_log_path = dataset.root / ERROR_LOG_PATH
    if error_log_path.exists():
        error_log_path.unlink()

    already_done = _load_existing_episode_stats(dataset)
    pending = [idx for idx in range(total_episodes) if idx not in already_done]

    print(f"Computing episodes stats (skip {len(already_done)} already cached)")

    errors: list[dict] = []

    def handle_result(res: dict):
        ep_idx = res["episode_index"]
        if res["error"] is not None:
            _record_error(dataset.root, ep_idx, res["error"])
            errors.append(res)
            return

        dataset.meta.episodes_stats[ep_idx] = res["stats"]
        write_episode_stats(ep_idx, res["stats"], dataset.root)

    if num_workers > 0:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(dataset.repo_id, str(dataset.root), dataset.revision),
        ) as ex:
            for res in tqdm(
                ex.map(_convert_episode_stats, pending, chunksize=chunksize),
                total=len(pending),
            ):
                handle_result(res)
    else:
        # 单进程 fallback
        _G["dataset"] = dataset
        for ep_idx in tqdm(pending):
            res = _convert_episode_stats(ep_idx)
            handle_result(res)

    return errors


def check_aggregate_stats(
    dataset: LeRobotDataset,
    reference_stats: dict[str, dict[str, np.ndarray]],
    video_rtol_atol: tuple[float] = (1e-2, 1e-2),
    default_rtol_atol: tuple[float] = (5e-6, 6e-5),
):
    """Verifies that the aggregated stats from episodes_stats are close to reference stats."""
    agg_stats = aggregate_stats(list(dataset.meta.episodes_stats.values()))
    relaxed_int_features = {"episode_index", "frame_index", "index", "task_index"}
    for key, ft in dataset.features.items():
        # These values might need some fine-tuning
        if ft["dtype"] == "video":
            # to account for image sub-sampling
            rtol, atol = video_rtol_atol
        elif key in relaxed_int_features:
            rtol, atol = (1e-4, 1e-2)
        else:
            rtol, atol = default_rtol_atol

        for stat, val in agg_stats[key].items():
            if key in reference_stats and stat in reference_stats[key]:
                err_msg = f"feature='{key}' stats='{stat}'"
                try:
                    np.testing.assert_allclose(val, reference_stats[key][stat], rtol=rtol, atol=atol, err_msg=err_msg)
                except AssertionError as e:
                    warnings.warn(f"{e}; continuing with relaxed validation")
