from pathlib import Path

EXPECTED_TOTAL = 4067.78


def test_total():
    assert Path("/app/out.txt").read_text().strip() == str(EXPECTED_TOTAL)
