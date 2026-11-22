# Copyright (c) OpenMMLab. All rights reserved.
"""PD disaggregation over the token-in/token-out ``/generate`` endpoint.

Prefill: ``with_cache``/``preserve_cache`` returns the cache handle like
``/v1/completions``. Decode: ``migration_request`` pulls it; its first output
token is the prefill's sampled token (no logprob, no local forward). Routed
experts stay indexed by absolute position on the migrated sequence. Only
request handling and sequence bookkeeping are exercised: no model or RDMA.
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import APIRouter

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.messages import SamplingParam, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.serve.openai.endpoints.generate import check_request, register
from lmdeploy.serve.openai.protocol import GenerateReqInput, GenerateReqOutput

MIGRATION = MigrationRequest(protocol=MigrationProtocol.RDMA,
                             remote_engine_id='http://p',
                             remote_session_id=5,
                             remote_token_id=7,
                             remote_block_ids=[0])


class _SessionManager:

    def has(self, session_id):
        return False

    def remove(self, session):
        pass


class _Context:

    def __init__(self, config, async_engine=None):
        self.async_engine = async_engine or SimpleNamespace(backend_config=config, speculative_config=None)
        self.session_manager = _SessionManager()
        self.default_gen_config = {}

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    def create_session(self, session_id):
        return SimpleNamespace(session_id=1234)


class _RawRequest:

    async def is_disconnected(self):
        return False


def _config():
    return PytorchEngineConfig(logprobs_mode='raw_logprobs', enable_return_routed_experts=True)


def test_disaggregation_request_validation():
    ctx = _Context(_config())
    ok = dict(input_ids=[1, 2, 3], max_tokens=1)
    assert check_request(GenerateReqInput(**ok, with_cache=True, preserve_cache=True), ctx) == ''
    assert check_request(GenerateReqInput(**ok, migration_request=MIGRATION), ctx) == ''
    assert check_request(
        GenerateReqInput(**ok, migration_request=MIGRATION, return_routed_experts=True,
                         routed_experts_start_len=3), ctx) == ''
    assert 'routed_experts_start_len=len(input_ids)' in check_request(
        GenerateReqInput(**ok, migration_request=MIGRATION, return_routed_experts=True), ctx)
    assert 'preserve_cache requires with_cache' in check_request(
        GenerateReqInput(**ok, preserve_cache=True), ctx)
    assert 'same request' in check_request(
        GenerateReqInput(**ok, with_cache=True, migration_request=MIGRATION), ctx)
    assert 'stream' in check_request(GenerateReqInput(**ok, with_cache=True, stream=True), ctx)
    assert 'input_ids' in check_request(GenerateReqInput(prompt='hi', with_cache=True), ctx)
    assert 'PyTorch engine' in check_request(
        GenerateReqInput(**ok, with_cache=True), _Context(TurbomindEngineConfig()))
    # the decode side may stream
    assert check_request(GenerateReqInput(**ok, migration_request=MIGRATION, stream=True), ctx) == ''
    # the JSON wire form validates into a MigrationRequest
    wire = GenerateReqInput.model_validate({**ok, 'migration_request': MIGRATION.model_dump(mode='json')})
    assert wire.migration_request == MIGRATION


class _Engine:
    """Records the generation config and replays scripted engine outputs."""

    def __init__(self, outputs):
        self.backend_config = _config()
        self.speculative_config = None
        self.outputs = outputs
        self.gen_config = None

    async def preprocess(self, **kwargs):
        self.gen_config = kwargs['gen_config']
        return SimpleNamespace()

    async def generate(self, prepared_request, stream_response=True):
        for out in self.outputs:
            base = dict(input_token_len=3, cache_block_ids=None, routed_experts=None, logprobs=None,
                        finish_reason=None)
            yield SimpleNamespace(**{**base, **out})


def _call(engine, request):
    router = APIRouter()
    register(router, _Context(engine.backend_config, async_engine=engine))
    endpoint = next(r.endpoint for r in router.routes if r.path == '/generate')
    return asyncio.run(endpoint(request, raw_request=_RawRequest()))


def test_prefill_returns_cache_handle_and_prompt_routes():
    routes = [[[1, 2]], [[3, 4]], [[5, 6]]]  # positions 0..2 (the whole prompt)
    engine = _Engine([dict(response='x', token_ids=[7], cache_block_ids=[4, 9], logprobs=[{7: -0.5}],
                           finish_reason='length', generate_token_len=1, routed_experts=routes)])
    result = _call(engine, GenerateReqInput(input_ids=[1, 2, 3], max_tokens=1, return_logprob=True,
                                            return_routed_experts=True, with_cache=True, preserve_cache=True))
    assert isinstance(result, GenerateReqOutput)
    assert engine.gen_config.with_cache and engine.gen_config.preserve_cache
    assert engine.gen_config.migration_request is None
    assert result.cache_session_id == 1234  # the engine session, not the request session_id
    assert result.cache_block_ids == [4, 9]
    assert result.remote_token_ids == [7]
    assert result.output_ids == [7]
    assert result.meta_info.output_token_logprobs == [(-0.5, 7)]
    assert result.meta_info.routed_experts == routes


def test_decode_skips_migrated_token_logprob():
    # The engine first emits the prefill's token (no logprob), then its own samples.
    engine = _Engine([
        dict(response='a', token_ids=[7], generate_token_len=1),
        dict(response='b', token_ids=[8], logprobs=[{8: -0.25}], generate_token_len=2),
        dict(response='c', token_ids=[9], logprobs=[{9: -0.75}], finish_reason='stop', generate_token_len=3,
             routed_experts=[[[1, 1]], [[2, 2]]]),
    ])
    result = _call(engine, GenerateReqInput(input_ids=[1, 2, 3], max_tokens=4, return_logprob=True,
                                            return_routed_experts=True, routed_experts_start_len=3,
                                            migration_request=MIGRATION))
    assert engine.gen_config.migration_request == MIGRATION
    assert result.output_ids == [7, 8, 9]
    assert result.meta_info.output_token_logprobs == [(-0.25, 8), (-0.75, 9)]
    assert result.meta_info.routed_experts == [[[1, 1]], [[2, 2]]]
    assert result.cache_session_id is None and result.cache_block_ids is None


def test_decode_stream_pairs_logprobs_when_migrated_token_shares_a_chunk():
    import json
    engine = _Engine([
        dict(response='ab', token_ids=[7, 8], logprobs=[{8: -0.25}], generate_token_len=2),
        dict(response='c', token_ids=[9], logprobs=[{9: -0.75}], finish_reason='length', generate_token_len=3),
    ])
    response = _call(engine, GenerateReqInput(input_ids=[1, 2, 3], max_tokens=3, return_logprob=True,
                                              migration_request=MIGRATION, stream=True))

    async def body():
        return [chunk async for chunk in response.body_iterator]

    chunks = [json.loads(c[6:]) for c in asyncio.run(body()) if c.startswith('data: {')]
    assert [c['output_ids'] for c in chunks] == [[7, 8], [9]]
    assert [c['meta_info']['output_token_logprobs'] for c in chunks] == [[[-0.25, 8]], [[-0.75, 9]]]


# ------------------------------------------------------------- sequence bookkeeping


def _sequence(migration_request=None, input_len=5):
    scheduler = Scheduler(
        SchedulerConfig(max_batches=4, max_session_len=128, max_request_output_len=64),
        CacheConfig(max_batches=4, block_size=16, num_cpu_blocks=0, num_gpu_blocks=16),
        SequenceMeta(16, strategy=ARSequenceStrategy()))
    session = scheduler.add_session(1)
    return session.add_sequence(np.arange(1, input_len + 1),
                                sampling_param=SamplingParam(return_routed_experts=True, max_new_tokens=8),
                                migration_request=migration_request)


def _rows(positions):
    return np.stack([np.full((2, 2), p + 1, dtype=np.uint16) for p in positions])


def test_unified_sequence_routes_are_one_row_per_computed_position():
    seq = _sequence()
    seq.update_token_ids(np.array([10]), mode=UpdateTokenMode.PREFILL, routed_experts=_rows(range(5)))
    seq.update_token_ids(np.array([11]), mode=UpdateTokenMode.DECODE, routed_experts=_rows([5]))
    assert seq.routed_experts[:, 0, 0].tolist() == [1, 2, 3, 4, 5, 6]


def test_migrated_sequence_routes_stay_absolute():
    """Decode of a 5-token prompt: the migrated token (pos 5) is first computed by
    the first local forward, so local rows start at position 5."""
    seq = _sequence(MIGRATION)
    # migration output: the prefill's token is appended without a forward
    seq.update_token_ids(np.array([7]), mode=UpdateTokenMode.PREFILL)
    assert len(seq.all_routed_experts) == 0
    # first local forward (scheduled as a 1-token prefill): input pos 5, samples pos 6
    seq.update_token_ids(np.array([8]), mode=UpdateTokenMode.PREFILL, routed_experts=_rows([5]))
    seq.update_token_ids(np.array([9]), mode=UpdateTokenMode.DECODE, routed_experts=_rows([6]))
    experts = seq.routed_experts
    assert experts.shape == (7, 2, 2)  # num_all_ids - 1: positions 0..6
    assert experts[:5].max() == 0  # prompt placeholders (prefill computed them)
    assert experts[5:, 0, 0].tolist() == [6, 7]
    # /generate returns rows from routed_experts_start_len = len(input_ids) = 5
    assert experts[5:].shape[0] == 2


def test_migrated_sequence_recompute_is_not_padded():
    """Preempted after migration: recompute prefill appends real rows from 0."""
    seq = _sequence(MIGRATION)
    seq.update_token_ids(np.array([7]), mode=UpdateTokenMode.PREFILL)
    seq.set_step(0)
    seq.update_token_ids(np.array([8]), mode=UpdateTokenMode.PREFILL, routed_experts=_rows(range(6)))
    assert seq.routed_experts[:, 0, 0].tolist() == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize('migrated', [False, True])
def test_trie_never_caches_migrated_prompt_placeholders(migrated):
    from lmdeploy.pytorch.paging.block_trie.trie import BlockTrie
    seq = _sequence(MIGRATION if migrated else None, input_len=20)
    seq.update_token_ids(np.array([7]), mode=UpdateTokenMode.PREFILL,
                         routed_experts=None if migrated else _rows(range(20)))
    if migrated:
        seq.update_token_ids(np.array([8]), mode=UpdateTokenMode.DECODE, routed_experts=_rows([20]))
    got = BlockTrie._get_routed_experts_for_range(seq, 0, 16)
    assert (got is None) == migrated
