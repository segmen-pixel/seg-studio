# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Labelling driven by a vision model, as part of the product.

The loop used to live beside the bridge in scripts/examples, which is why the
trainer API could not offer it: an example cannot be imported by the thing it
is an example of. It runs here now, and the bridge stays a subprocess -- so a
mask written by a model still arrives at the API as a request, with the header
that records who wrote it and the guard that refuses to paint over a person.

`fastmcp` is imported lazily by the loop, and stays an optional dependency:
the feature answers 503 where it is not installed rather than making every
installation carry it.
"""
from .backends import ALIASES, Backend, Reply, describe, make_backend

__all__ = ["ALIASES", "Backend", "Reply", "describe", "make_backend"]
