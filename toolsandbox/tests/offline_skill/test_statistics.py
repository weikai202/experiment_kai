from toolsandbox_pipeline.offline.skill_statistics import rewrite_triggered
from toolsandbox_pipeline.schemas.skill import SkillOnlineStatistics


def stats(uses, failures, last=0):
    successes = uses - failures
    return SkillOnlineStatistics(
        evaluated_uses=uses,
        successes=successes,
        failures=failures,
        success_rate=successes / uses if uses else 0.0,
        last_update_attempt_at_use_count=last,
    )


def test_rewrite_threshold_is_exact_integer_and_attempt_gated():
    assert not rewrite_triggered(stats(9, 9))
    assert not rewrite_triggered(stats(12, 3))
    assert rewrite_triggered(stats(12, 4))
    assert not rewrite_triggered(stats(12, 4, last=12))
