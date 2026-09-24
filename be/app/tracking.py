from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class Track:
    id: int
    landmarks: list
    frames: deque = field(default_factory=lambda: deque(maxlen=10))
    label_index: int | None = None
    probabilities: list | None = None


def center(pose):
    return ((pose[23]['x'] + pose[24]['x']) / 2,
            (pose[23]['y'] + pose[24]['y']) / 2)


class SessionTracker:
    """Conservative spatial tracking, not persistent person identification."""
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks = []

    def update(self, poses):
        visible = []
        for pose in poses[:2]:
            try:
                valid = len(pose) == 33 and all(
                    math.isfinite(p[key]) for p in pose for key in ('x', 'y', 'z', 'visibility'))
            except (KeyError, TypeError):
                valid = False
            if not valid:
                raise ValueError('Expected 33 finite landmarks')
            if min(pose[i]['visibility'] for i in (11, 12, 23, 24)) >= .4:
                visible.append(pose)
        distances = [[math.dist(center(p), center(t.landmarks)) for t in self.tracks] for p in visible]
        candidates = []
        for row in distances:
            ranked = sorted(range(len(row)), key=row.__getitem__)
            match = ranked[0] if ranked and row[ranked[0]] < .2 else None
            if len(ranked) > 1 and row[ranked[1]] - row[ranked[0]] < .06:
                match = None
            candidates.append(match)
        current = []
        for pose, match in zip(visible, candidates):
            if match is not None and candidates.count(match) == 1:
                track = self.tracks[match]
            else:
                track = Track(self.next_id, pose)
                self.next_id += 1
            track.landmarks = pose
            track.frames.append([p[k] for p in pose for k in ('x', 'y', 'z', 'visibility')])
            current.append(track)
        self.tracks = current
        return current
