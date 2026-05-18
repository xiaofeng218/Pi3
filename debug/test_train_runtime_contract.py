from __future__ import annotations

import os
import sys
import unittest
from unittest import mock


class TrainRuntimeContractTests(unittest.TestCase):
    def test_apply_mp_sharing_strategy_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"PI3_MP_SHARING_STRATEGY": "file_system"}, clear=False):
            if "scripts.train_pi3x" in sys.modules:
                del sys.modules["scripts.train_pi3x"]
            with mock.patch("torch.multiprocessing.set_sharing_strategy") as set_strategy:
                import scripts.train_pi3x as train_pi3x

                train_pi3x._apply_mp_runtime_overrides()

        set_strategy.assert_called_once_with("file_system")


if __name__ == "__main__":
    unittest.main()
