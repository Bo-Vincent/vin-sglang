from types import SimpleNamespace

import pytest

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ModelDraftAlgorithm:
    @staticmethod
    def is_none() -> bool:
        return False

    @staticmethod
    def is_ngram() -> bool:
        return False


def test_draft_worker_rejects_inherited_remote_instance_loader() -> None:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.spec_algorithm = _ModelDraftAlgorithm()
    scheduler.server_args = SimpleNamespace(
        load_format="remote_instance",
        speculative_draft_load_format=None,
    )

    with pytest.raises(RuntimeError, match="speculative draft"):
        scheduler.maybe_init_draft_worker()
