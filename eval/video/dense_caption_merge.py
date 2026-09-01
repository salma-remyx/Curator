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

"""Build temporally dense captions from per-window clip captions.

Adapts the recursive clip-caption merging of LVD-2M (arXiv:2410.10816):
captions written for independent temporal units are merged pairwise and
recursively into a single caption covering the whole span, while every unit
keeps its own timestamp. The paper's VLM merge operator is substituted with a
parameter-free merge that drops text repeated at the seam between adjacent
units -- pass a callable as ``merge_operator`` to plug an LLM merger back in.
The paper's dataset-scale long-take filtering is out of scope here; pair this
with the pipeline's shot-aware splitting (``--splitting-algorithm transnetv2``)
for cut-free clips.

Reads a captioning pipeline output directory (``metas/v0/<uid>.json``) and
writes a new pipeline-output-shaped directory:

  * ``metas/v0/<uid>.json`` -- the original clip metadata with ``windows``
    collapsed to a single window whose caption is the dense merge. The result
    can be scored alongside the original run by ``caption_clipscore.py``:
    ``--caption-dirs qwen25=... qwen25_dense=<output-dir>`` compares dense vs
    clip-level captions directly.
  * ``dense/<source-video>.json`` (``--cross-clip``) -- per-source-video dense
    captions with absolute timestamps, one file per source video.

Example:

    python dense_caption_merge.py \\
        --input-dir /path/to/captions_qwen25 \\
        --output-dir /path/to/captions_qwen25_dense
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

# Merges the caption of two adjacent temporal units into one string.
CaptionMerger = Callable[[str, str], str]


@dataclass(frozen=True)
class CaptionSegment:
    """A caption together with the time span it covers, in seconds."""

    start_s: float
    end_s: float
    text: str


def default_merge_operator(left: str, right: str) -> str:
    """Merge two caption strings, dropping words repeated at the seam.

    Sliding-window captioners restate the tail of one window at the head of
    the next; trimming the longest such overlap keeps a recursive merge from
    amplifying that repetition level after level.
    """
    left_words = left.split()
    right_words = right.split()
    for overlap in range(min(len(left_words), len(right_words)), 0, -1):
        if left_words[-overlap:] == right_words[:overlap]:
            return " ".join(left_words + right_words[overlap:])
    return " ".join(left_words + right_words)


def merge_levels(
    segments: list[CaptionSegment],
    merge_operator: CaptionMerger = default_merge_operator,
) -> list[list[CaptionSegment]]:
    """Recursively merge adjacent segments pairwise, one level at a time.

    Returns every granularity level, leaves first and the single full-span
    caption last: ``levels[-1][0]`` merges the whole span, and each level
    halves the number of captions so a bounded-context merge operator never
    sees more than two captions at a time.
    """
    levels = [sorted(segments, key=lambda segment: segment.start_s)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        merged = [
            CaptionSegment(
                current[i].start_s,
                current[i + 1].end_s,
                merge_operator(current[i].text, current[i + 1].text),
            )
            for i in range(0, len(current) - 1, 2)
        ]
        if len(current) % 2:
            merged.append(current[-1])
        levels.append(merged)
    return levels


def merge_dense(
    segments: list[CaptionSegment],
    merge_operator: CaptionMerger = default_merge_operator,
) -> CaptionSegment | None:
    """Return the single dense caption covering all ``segments``, or None."""
    if not segments:
        return None
    return merge_levels(segments, merge_operator)[-1][0]


@dataclass
class ClipCaptions:
    """Caption segments parsed from one clip's metadata, grouped by caption key."""

    source_video: str
    span: tuple[float, float]
    segments_by_key: dict[str, list[CaptionSegment]]
    frame_bounds: tuple[int | None, int | None]

    @property
    def primary_key(self) -> str | None:
        """Caption key that ``caption_clipscore`` scores first, or None."""
        return next(iter(self.segments_by_key), None)


def _caption_entries(window: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield each caption entry of a window, first-key-first.

    Follows ``caption_clipscore._get_window_captions``: any string value whose
    key contains "caption" counts, in insertion order.
    """
    for key, value in window.items():
        if "caption" in key and isinstance(value, str) and value.strip():
            yield key, value.strip()


def _frame_span(window: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return the (start, end) frame bounds of a window, if present."""
    start = window.get("start_frame")
    end = window.get("end_frame")
    return (
        start if isinstance(start, int) else None,
        end if isinstance(end, int) else None,
    )


def _to_segments(
    entries: list[tuple[tuple[int | None, int | None], str]],
    span: tuple[float, float],
    framerate: float,
) -> list[CaptionSegment]:
    """Convert (frame bounds, caption) entries to absolute-second segments.

    Frame bounds are used when the metadata carries them; otherwise the clip's
    duration span is spread evenly across its captioned windows.
    """
    if framerate > 0 and all(start is not None and end is not None for (start, end), _ in entries):
        return [
            CaptionSegment(span[0] + start / framerate, span[0] + end / framerate, text)
            for (start, end), text in entries
        ]
    step = max(span[1] - span[0], 0.0) / len(entries) if entries else 0.0
    return [CaptionSegment(span[0] + i * step, span[0] + (i + 1) * step, text) for i, (_, text) in enumerate(entries)]


def parse_clip_captions(data: dict[str, Any]) -> ClipCaptions:
    """Parse a clip metadata JSON into per-key caption segments."""
    span_raw = data.get("duration_span") or [0.0, 0.0]
    span = (float(span_raw[0]), float(span_raw[1]))
    framerate = float(data.get("framerate_source") or 0.0)

    entries: dict[str, list[tuple[tuple[int | None, int | None], str]]] = {}
    frame_bounds: list[int | None] = [None, None]
    for window in data.get("windows") or []:
        window_entries = list(_caption_entries(window))
        if not window_entries:
            continue
        start, end = _frame_span(window)
        if frame_bounds[0] is None and start is not None:
            frame_bounds[0] = start
        if end is not None:
            frame_bounds[1] = end
        for key, text in window_entries:
            entries.setdefault(key, []).append(((start, end), text))

    segments_by_key = {key: _to_segments(values, span, framerate) for key, values in entries.items()}
    source = str(data.get("source_video") or data.get("video_path") or "unknown")
    return ClipCaptions(
        source_video=source,
        span=span,
        segments_by_key=segments_by_key,
        frame_bounds=(frame_bounds[0], frame_bounds[1]),
    )


def _write_cross_clip_captions(
    clips_by_source: dict[str, list[tuple[tuple[float, float], str]]],
    dense_dir: Path,
    merge_operator: CaptionMerger,
) -> tuple[int, float]:
    """Merge each source video's per-clip dense captions into one caption file."""
    dense_dir.mkdir(parents=True, exist_ok=True)
    sources = 0
    covered_s = 0.0
    for source, clips in sorted(clips_by_source.items()):
        clips.sort(key=lambda clip: clip[0][0])
        segments = [CaptionSegment(start, end, text) for (start, end), text in clips]
        dense = merge_dense(segments, merge_operator)
        if dense is None:
            continue
        payload = {
            "source_video": source,
            "text": dense.text,
            "segments": [asdict(segment) for segment in segments],
            "clip_count": len(segments),
        }
        (dense_dir / f"{Path(source).stem}.json").write_text(json.dumps(payload, indent=2))
        sources += 1
        covered_s += segments[-1].end_s - segments[0].start_s
    return sources, covered_s


def build_dense_captions(
    input_dir: str,
    output_dir: str,
    cross_clip: bool = False,
    merge_operator: CaptionMerger = default_merge_operator,
) -> dict[str, int | float]:
    """Merge per-window captions into dense captions, in pipeline-output shape.

    Writes ``<output_dir>/metas/v0/<uid>.json`` for every clip in ``input_dir``
    with ``windows`` collapsed to the dense merge, so the result scores like any
    other captioning run. With ``cross_clip``, also writes per-source-video
    dense captions to ``<output_dir>/dense/``.
    """
    meta_dir = Path(input_dir) / "metas" / "v0"
    if not meta_dir.is_dir():
        msg = f"No clip metadata directory at {meta_dir}"
        raise FileNotFoundError(msg)
    out_meta_dir = Path(output_dir) / "metas" / "v0"
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int | float] = {
        "clips": 0,
        "clips_with_captions": 0,
        "dense_sources": 0,
        "covered_s": 0.0,
    }
    clips_by_source: dict[str, list[tuple[tuple[float, float], str]]] = {}

    for meta_path in sorted(meta_dir.glob("*.json")):
        data: dict[str, Any] = json.loads(meta_path.read_text())
        clip = parse_clip_captions(data)
        summary["clips"] += 1
        if clip.segments_by_key:
            summary["clips_with_captions"] += 1

            dense_by_key = {
                key: merge_dense(segments, merge_operator) for key, segments in clip.segments_by_key.items()
            }
            window: dict[str, Any] = {}
            if clip.frame_bounds[0] is not None:
                window["start_frame"] = clip.frame_bounds[0]
            if clip.frame_bounds[1] is not None:
                window["end_frame"] = clip.frame_bounds[1]
            for key, dense in dense_by_key.items():
                window[key] = dense.text if dense else ""

            primary_key = clip.primary_key
            primary_segments = clip.segments_by_key[primary_key]
            data["windows"] = [window]
            data["dense_caption"] = {
                "caption_key": primary_key,
                "text": window[primary_key],
                "segments": [asdict(segment) for segment in primary_segments],
            }
            clips_by_source.setdefault(clip.source_video, []).append((clip.span, window[primary_key]))

        # Captionless clips pass through unchanged so UID sets stay comparable
        # when scoring dense and original caption directories side by side.
        (out_meta_dir / meta_path.name).write_text(json.dumps(data, indent=2))

    if cross_clip and clips_by_source:
        summary["dense_sources"], summary["covered_s"] = _write_cross_clip_captions(
            clips_by_source, Path(output_dir) / "dense", merge_operator
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build temporally dense captions from per-window clip captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Captioning pipeline output directory containing metas/v0/<uid>.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the dense-caption pipeline output to (created if missing).",
    )
    parser.add_argument(
        "--cross-clip",
        action="store_true",
        help="Also merge per-clip dense captions into per-source-video dense captions "
        "(written to <output-dir>/dense/<source-video>.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_dense_captions(args.input_dir, args.output_dir, cross_clip=args.cross_clip)
    logger.info(f"Dense captions for {summary['clips_with_captions']}/{summary['clips']} clips -> {args.output_dir}")
    if summary["dense_sources"]:
        logger.info(f"  Sources: {summary['dense_sources']} ({summary['covered_s']:.1f}s covered)")


if __name__ == "__main__":
    main()
