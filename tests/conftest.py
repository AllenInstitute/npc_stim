import gc

import pytest


@pytest.fixture(autouse=True)
def _collect_garbage_after_test():
    """h5py objects opened over fsspec (S3) left for the garbage collector can be finalized on
    the fsspec thread during a later test's read, which hangs the suite: collect them here."""
    yield
    gc.collect()
