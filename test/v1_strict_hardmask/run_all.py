import sys
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent

# V1 전용 테스트만 탐색해 저장소의 다른 legacy 테스트와 분리 실행함.
suite = unittest.defaultTestLoader.discover(str(TEST_DIR), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
