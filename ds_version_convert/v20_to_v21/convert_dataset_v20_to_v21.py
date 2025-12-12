from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

from lerobot.datasets.compute_stats import get_feature_stats, sample_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_episode_stats


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

        axes_to_reduce = (0, 2, 3) if ft["dtype"] in ["image", "video"] else 0
        keepdims = True if ft["dtype"] in ["image", "video"] else (ep_ft_data.ndim == 1)

        ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)

        if ft["dtype"] in ["image", "video"]:
            # remove batch dim
            ep_stats[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()}

    return ep_stats, ep_idx


def convert_stats(dataset: LeRobotDataset, num_workers: int = 8, chunksize: int = 4):
    assert dataset.episodes is None
    print("Computing episodes stats")

    total_episodes = dataset.meta.total_episodes

    if num_workers > 0:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(dataset.repo_id, str(dataset.root), dataset.revision),
        ) as ex:
            # ✅ map + chunksize：调度开销更低
            for ep_stats, ep_idx in tqdm(
                ex.map(_convert_episode_stats, range(total_episodes), chunksize=chunksize),
                total=total_episodes,
            ):
                dataset.meta.episodes_stats[ep_idx] = ep_stats
    else:
        # 单进程 fallback
        _G["dataset"] = dataset
        for ep_idx in tqdm(range(total_episodes)):
            ep_stats, _ = _convert_episode_stats(ep_idx)
            dataset.meta.episodes_stats[ep_idx] = ep_stats

    # 写文件建议保持串行（避免 NFS/小文件并发抖动）
    for ep_idx in tqdm(range(total_episodes)):
        write_episode_stats(ep_idx, dataset.meta.episodes_stats[ep_idx], dataset.root)


def check_aggregate_stats(
    dataset: LeRobotDataset,
    reference_stats: dict[str, dict[str, np.ndarray]],
    video_rtol_atol: tuple[float] = (1e-2, 1e-2),
    default_rtol_atol: tuple[float] = (5e-6, 6e-5),
):
    """Verifies that the aggregated stats from episodes_stats are close to reference stats."""
    agg_stats = aggregate_stats(list(dataset.meta.episodes_stats.values()))
    for key, ft in dataset.features.items():
        # These values might need some fine-tuning
        if ft["dtype"] == "video":
            # to account for image sub-sampling
            rtol, atol = video_rtol_atol
        else:
            rtol, atol = default_rtol_atol

        for stat, val in agg_stats[key].items():
            if key in reference_stats and stat in reference_stats[key]:
                err_msg = f"feature='{key}' stats='{stat}'"
                np.testing.assert_allclose(val, reference_stats[key][stat], rtol=rtol, atol=atol, err_msg=err_msg)
