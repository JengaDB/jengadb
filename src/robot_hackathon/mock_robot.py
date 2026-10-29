from datetime import datetime, timedelta, timezone

class MockRobot:
    """Synthetic observations only. No hardware commands or network clients."""
    def attempts(self):
        start = datetime(2026, 10, 29, 17, tzinfo=timezone.utc)
        outcomes = ("success", "success", "failure", "success", "unknown")
        return [dict(schema_version=1, attempt_id=f"mock-v1-{i+1:03d}",
                     run_id="mock-v1", timestamp=(start + timedelta(seconds=i)).isoformat(),
                     mode="mock", machine_alias="synthetic-only", object_label="foam-cube",
                     action_type="simulated-placement", strategy="baseline" if i < 10 else "centered",
                     pose=None, observed_outcome=outcomes[i % 5], duration_ms=800 + i * 25,
                     error_class="unobserved" if i % 5 == 4 else None,
                     confidence=None, artifact_refs=[]) for i in range(20)]
