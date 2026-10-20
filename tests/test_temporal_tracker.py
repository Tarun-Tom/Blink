import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.webcam_detector import (
    TemporalDrowsinessTracker,
    STATE_OPEN,
    STATE_CLOSED,
    STATE_UNKNOWN
)

def test_temporal_tracker():
    print("=" * 65)
    print("RUNNING TEMPORAL TRACKER SYNTHETIC SEQUENCE TESTS")
    print("=" * 65)

    fps = 30.0
    dt = 1.0 / fps # ~0.0333s

    # TEST 1: Stable CLOSED -> Must trigger alert after 1.5s
    print("\n[Test 1] Synthetic Stable CLOSED sequence...")
    tracker = TemporalDrowsinessTracker(
        alert_threshold=1.5,
        open_grace_period=0.25,
        unknown_timeout=0.5,
        window_size=5,
        decision_threshold=0.5
    )

    t = 0.0
    alert_triggered_at = None
    for frame in range(60): # 2.0 seconds
        res = tracker.update(raw_p_awake=0.45, current_time=t) # Closed (P < 0.5)
        if res["is_alert"] and alert_triggered_at is None:
            alert_triggered_at = t
        t += dt

    assert alert_triggered_at is not None, "Alert was never triggered on stable closed sequence!"
    assert 1.45 <= alert_triggered_at <= 1.65, f"Alert triggered at unexpected time: {alert_triggered_at:.3f}s"
    print(f"  PASS: Stable CLOSED triggered alert at t={alert_triggered_at:.3f}s (Threshold: 1.5s)")

    # TEST 2: Brief OPEN flickers (<250ms) during CLOSED -> Must NOT reset closure duration
    print("\n[Test 2] Synthetic 1-frame OPEN flickers during closed state...")
    tracker.reset()
    t = 0.0
    # 40 frames closed (~1.33s)
    for _ in range(40):
        tracker.update(raw_p_awake=0.48, current_time=t)
        t += dt

    closure_before_flicker = tracker.closure_duration
    assert closure_before_flicker >= 1.25, f"Expected ~1.3s closure, got {closure_before_flicker:.3f}s"

    # Single-frame OPEN flicker (dt ~ 33ms < 250ms grace period)
    flicker_res = tracker.update(raw_p_awake=0.90, current_time=t)
    t += dt
    assert flicker_res["state"] == STATE_OPEN or flicker_res["state"] == STATE_CLOSED
    assert tracker.closure_duration >= closure_before_flicker, (
        f"Flicker wiped closure duration! Before: {closure_before_flicker:.3f}s, After: {tracker.closure_duration:.3f}s"
    )

    # 10 more frames closed (~0.33s) -> Should push total past 1.5s and trigger alert!
    alert_triggered = False
    for _ in range(10):
        res = tracker.update(raw_p_awake=0.48, current_time=t)
        if res["is_alert"]:
            alert_triggered = True
        t += dt

    assert alert_triggered, "Alert failed to trigger despite closure resuming after flicker!"
    print(f"  PASS: 1-frame OPEN flicker preserved closure duration ({tracker.closure_duration:.2f}s) and successfully triggered alert.")

    # TEST 3: Sustained OPEN (>250ms) -> Must reset closure duration to 0.0s
    print("\n[Test 3] Synthetic Sustained OPEN sequence (>250ms)...")
    tracker.reset()
    t = 0.0
    # 40 frames closed (~1.33s)
    for _ in range(40):
        tracker.update(raw_p_awake=0.48, current_time=t)
        t += dt

    # 12 frames OPEN (~0.4s > 0.25s grace period)
    for _ in range(12):
        tracker.update(raw_p_awake=0.95, current_time=t)
        t += dt

    assert tracker.closure_duration == 0.0, f"Sustained OPEN failed to reset closure duration: {tracker.closure_duration}"
    assert tracker.open_start_time is None, "open_start_time was not cleared upon reset"
    print("  PASS: Sustained OPEN cleanly reset closure duration to 0.0s.")

    # TEST 4: UNKNOWN interruptions -> Must not count as closure; prolonged UNKNOWN resets
    print("\n[Test 4] Synthetic UNKNOWN interruptions (detection dropout)...")
    tracker.reset()
    t = 0.0
    # 30 frames closed (~1.0s)
    for _ in range(30):
        tracker.update(raw_p_awake=0.48, current_time=t)
        t += dt

    closure_at_1s = tracker.closure_duration

    # 2 frames UNKNOWN (dropout ~66ms < 500ms timeout)
    for _ in range(2):
        res = tracker.update(raw_p_awake=None, current_time=t)
        assert res["state"] == STATE_UNKNOWN
        t += dt

    # UNKNOWN should NOT have added to closure duration!
    assert tracker.closure_duration == closure_at_1s, (
        f"UNKNOWN falsely accumulated closure duration! Before: {closure_at_1s:.3f}s, After: {tracker.closure_duration:.3f}s"
    )

    # Now prolonged UNKNOWN (20 frames = 0.66s > 0.5s timeout)
    for _ in range(20):
        tracker.update(raw_p_awake=None, current_time=t)
        t += dt

    assert tracker.closure_duration == 0.0, f"Prolonged UNKNOWN failed to reset closure duration: {tracker.closure_duration}"
    print("  PASS: UNKNOWN never accumulated closure; brief dropout held state, prolonged dropout reset duration.")

    print("\n" + "=" * 65)
    print("ALL SYNTHETIC SEQUENCE TESTS PASSED [4/4]")
    print("=" * 65)
    return True

if __name__ == "__main__":
    test_temporal_tracker()
