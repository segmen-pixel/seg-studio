# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import pytest


@pytest.fixture(autouse=True)
def _chat_state_is_the_tests_own(tmp_path_factory, monkeypatch):
    """Keep the example chat's saved conversations and connection out of $HOME."""
    monkeypatch.setenv("SEG_VLM_CHAT_STATE", str(tmp_path_factory.mktemp("vlm-chat")))
