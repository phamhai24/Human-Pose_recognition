import pytest
from be.app.tracking import SessionTracker


def pose(x, visibility=1.0):
    return [dict(x=x, y=.5, z=0., visibility=visibility) for _ in range(33)]


def test_reordered_people_keep_separate_sequences():
    tracker = SessionTracker()
    first = tracker.update([pose(.2), pose(.8)])
    ids = [t.id for t in first]
    second = tracker.update([pose(.8), pose(.2)])
    assert [t.id for t in second] == ids[::-1]
    assert second[0].frames[0][0] == .8
    assert second[1].frames[0][0] == .2


def test_lost_person_does_not_leave_stale_result():
    tracker = SessionTracker()
    old = tracker.update([pose(.2)])[0].id
    assert tracker.update([]) == []
    current = tracker.update([pose(.2)])[0]
    assert current.id != old
    assert len(current.frames) == 1


def test_sessions_and_buffer_limits():
    a, b = SessionTracker(), SessionTracker()
    for _ in range(15):
        result = a.update([pose(.2)])
    assert len(result[0].frames) == 10
    assert len(b.update([pose(.2)])[0].frames) == 1


def test_ambiguous_crossing_resets_sequences():
    tracker = SessionTracker()
    tracker.update([pose(.4), pose(.6)])
    result = tracker.update([pose(.49), pose(.51)])
    assert all(len(t.frames) == 1 for t in result)


def test_invisible_pose_does_not_enter_model_buffer():
    assert SessionTracker().update([pose(.5, 0)]) == []


def test_malformed_landmarks_rejected():
    with pytest.raises(ValueError):
        SessionTracker().update([[{'x': .5}]])
