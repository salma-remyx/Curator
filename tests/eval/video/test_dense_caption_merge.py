# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for dense_caption_merge.py helper functions (CPU only)."""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from eval.video.caption_clipscore import _collect_tasks, _get_window_captions
from eval.video.dense_caption_merge import (
    CaptionSegment,
    build_dense_captions,
    default_merge_operator,
    merge_dense,
    merge_levels,
    parse_clip_captions,
)


@pytest.fixture
def tmp_dir() -> Generator[str]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def caption_run(tmp_dir: str) -> str:
    """Create a captioning pipeline output directory with four clips.

    ``uid-aaa`` and ``uid-bbb`` share a source video (contiguous spans), as do
    ``uid-ccc`` and ``uid-empty``; ``uid-empty`` carries no captions. The two
    windows of ``uid-aaa`` repeat "a sunlit field." at the seam.
    """
    meta_dir = os.path.join(tmp_dir, "captions_qwen25", "metas", "v0")
    os.makedirs(meta_dir)
    metas = [
        {
            "source_video": "/data/sourceA.mp4",
            "duration_span": [0.0, 10.0],
            "framerate_source": 24.0,
            "windows": [
                {"start_frame": 0, "end_frame": 120, "qwen2.5_caption": "A dog runs across a sunlit field."},
                {"start_frame": 120, "end_frame": 240, "qwen2.5_caption": "a sunlit field. The dog barks loudly."},
            ],
        },
        {
            "source_video": "/data/sourceA.mp4",
            "duration_span": [10.0, 20.0],
            "framerate_source": 24.0,
            "windows": [{"start_frame": 0, "end_frame": 240, "qwen2.5_caption": "Waves crash on a rocky shore."}],
        },
        {
            "source_video": "/data/sourceB.mp4",
            "duration_span": [0.0, 8.0],
            "framerate_source": 24.0,
            "windows": [
                {"start_frame": 0, "end_frame": 192, "qwen2.5_caption": "A cyclist pedals up a mountain road."},
            ],
        },
        {
            "source_video": "/data/sourceB.mp4",
            "duration_span": [8.0, 16.0],
            "framerate_source": 24.0,
            "windows": [],
        },
    ]
    uids = ["uid-aaa", "uid-bbb", "uid-ccc", "uid-empty"]
    for uid, meta in zip(uids, metas, strict=True):
        with open(os.path.join(meta_dir, f"{uid}.json"), "w") as f:
            json.dump(meta, f)
    return os.path.join(tmp_dir, "captions_qwen25")


@pytest.fixture
def dense_out(tmp_dir: str) -> str:
    """Output directory path for build_dense_captions runs."""
    return os.path.join(tmp_dir, "captions_qwen25_dense")


def _segments(*texts: str) -> list[CaptionSegment]:
    """Build evenly spaced segments, one per text."""
    return [CaptionSegment(float(i), float(i + 1), text) for i, text in enumerate(texts)]


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---- default_merge_operator ----


class TestDefaultMergeOperator:
    def test_trims_seam_repetition(self) -> None:
        merged = default_merge_operator("A dog runs across a sunlit field.", "a sunlit field. The dog barks.")
        assert merged == "A dog runs across a sunlit field. The dog barks."

    def test_no_overlap_joins_with_space(self) -> None:
        assert default_merge_operator("One scene.", "Another scene.") == "One scene. Another scene."

    def test_identical_captions_collapse(self) -> None:
        assert default_merge_operator("The dog sits.", "The dog sits.") == "The dog sits."


# ---- merge_levels / merge_dense ----


class TestMergeLevels:
    def test_multi_granularity_levels(self) -> None:
        levels = merge_levels(_segments("a", "b", "c", "d"))
        assert [len(level) for level in levels] == [4, 2, 1]

    def test_odd_count_carries_last_segment(self) -> None:
        levels = merge_levels(_segments("a", "b", "c"))
        assert [len(level) for level in levels] == [3, 2, 1]
        assert levels[1][-1].text == "c"

    def test_sorted_by_start_time(self) -> None:
        levels = merge_levels([CaptionSegment(5.0, 6.0, "late"), CaptionSegment(0.0, 1.0, "early")])
        assert levels[-1][0].text == "early late"


class TestMergeDense:
    def test_empty_returns_none(self) -> None:
        assert merge_dense([]) is None

    def test_single_segment_passthrough(self) -> None:
        segment = CaptionSegment(0.0, 4.0, "only")
        assert merge_dense([segment]) == segment

    def test_covers_full_span(self) -> None:
        dense = merge_dense(_segments("a", "b", "c"))
        assert dense is not None
        assert dense.start_s == 0.0
        assert dense.end_s == 3.0
        assert dense.text == "a b c"


# ---- parse_clip_captions ----


class TestParseClipCaptions:
    def test_frame_bounds_become_absolute_seconds(self) -> None:
        data = {
            "duration_span": [10.0, 20.0],
            "framerate_source": 24.0,
            "windows": [
                {"start_frame": 24, "end_frame": 48, "m_caption": "first"},
                {"start_frame": 48, "end_frame": 96, "m_caption": "second"},
            ],
        }
        segments = parse_clip_captions(data).segments_by_key["m_caption"]
        assert [(s.start_s, s.end_s) for s in segments] == [(11.0, 12.0), (12.0, 14.0)]

    def test_spread_evenly_without_frame_metadata(self) -> None:
        data = {
            "duration_span": [10.0, 20.0],
            "windows": [{"m_caption": "first"}, {"m_caption": "second"}],
        }
        segments = parse_clip_captions(data).segments_by_key["m_caption"]
        assert [(s.start_s, s.end_s) for s in segments] == [(10.0, 15.0), (15.0, 20.0)]

    def test_source_video_fallback_and_primary_key(self) -> None:
        clip = parse_clip_captions({"video_path": "/fallback.mp4", "windows": []})
        assert clip.source_video == "/fallback.mp4"
        assert clip.primary_key is None

    def test_empty_caption_values_skipped(self) -> None:
        data = {"windows": [{"m_caption": "   "}, {"m_caption": "kept"}]}
        segments = parse_clip_captions(data).segments_by_key["m_caption"]
        assert [s.text for s in segments] == ["kept"]


# ---- build_dense_captions (integration with caption_clipscore) ----


class TestBuildDenseCaptions:
    def test_dense_windows_readable_by_caption_clipscore(self, caption_run: str, dense_out: str) -> None:
        build_dense_captions(caption_run, dense_out)
        captions = _get_window_captions(os.path.join(dense_out, "metas", "v0", "uid-aaa.json"))
        assert len(captions) == 1
        assert captions[0].count("sunlit field.") == 1
        assert captions[0].startswith("A dog runs")
        assert "barks" in captions[0]

    def test_collect_tasks_compares_dense_vs_clip_level(self, caption_run: str, dense_out: str) -> None:
        build_dense_captions(caption_run, dense_out)
        tasks = _collect_tasks(["uid-aaa"], {"clip": caption_run, "dense": dense_out})
        texts = {label: text for _uid, label, text in tasks}
        # The clip-level baseline joins window captions verbatim, duplicating the seam.
        assert texts["clip"].count("sunlit field.") == 2
        assert texts["dense"].count("sunlit field.") == 1

    def test_preserves_uid_resolution_keys(self, caption_run: str, dense_out: str) -> None:
        build_dense_captions(caption_run, dense_out)
        original = _load_json(os.path.join(caption_run, "metas", "v0", "uid-bbb.json"))
        dense = _load_json(os.path.join(dense_out, "metas", "v0", "uid-bbb.json"))
        assert dense["source_video"] == original["source_video"]
        assert dense["duration_span"] == original["duration_span"]

    def test_dense_caption_segments_are_timestamped(self, caption_run: str, dense_out: str) -> None:
        build_dense_captions(caption_run, dense_out)
        dense = _load_json(os.path.join(dense_out, "metas", "v0", "uid-aaa.json"))
        segments = dense["dense_caption"]["segments"]
        assert [s["start_s"] for s in segments] == [0.0, 5.0]
        assert [s["end_s"] for s in segments] == [5.0, 10.0]
        assert dense["windows"][0]["start_frame"] == 0
        assert dense["windows"][0]["end_frame"] == 240

    def test_clip_without_captions_passed_through(self, caption_run: str, dense_out: str) -> None:
        summary = build_dense_captions(caption_run, dense_out)
        assert summary["clips"] == 4
        assert summary["clips_with_captions"] == 3
        dense = _load_json(os.path.join(dense_out, "metas", "v0", "uid-empty.json"))
        assert dense["windows"] == []
        assert "dense_caption" not in dense

    def test_cross_clip_writes_per_source_files(self, caption_run: str, dense_out: str) -> None:
        summary = build_dense_captions(caption_run, dense_out, cross_clip=True)
        assert summary["dense_sources"] == 2
        assert summary["covered_s"] == pytest.approx(28.0)

        source_a = _load_json(os.path.join(dense_out, "dense", "sourceA.json"))
        assert source_a["clip_count"] == 2
        assert source_a["segments"][0]["start_s"] == pytest.approx(0.0)
        assert source_a["segments"][-1]["end_s"] == pytest.approx(20.0)
        assert "sunlit field" in source_a["text"]
        assert "rocky shore" in source_a["text"]

        source_b = _load_json(os.path.join(dense_out, "dense", "sourceB.json"))
        assert source_b["clip_count"] == 1
        assert "mountain road" in source_b["text"]

    def test_missing_meta_dir_raises(self, tmp_dir: str, dense_out: str) -> None:
        with pytest.raises(FileNotFoundError, match="No clip metadata directory"):
            build_dense_captions(os.path.join(tmp_dir, "missing"), dense_out)
