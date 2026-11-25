# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import base64
import json
from contextlib import aclosing
from http import HTTPStatus

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from lmdeploy.messages import LogprobArrays, PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.endpoints.common import build_serving_generation_config, validate_request
from lmdeploy.serve.openai.errors import (
    create_error_response,
    create_request_error_response,
    request_error_payload,
)
from lmdeploy.serve.openai.protocol import (
    GenerateReqFlatMetaOutput,
    GenerateReqInput,
    GenerateReqMetaOutput,
    GenerateReqOutput,
)
from lmdeploy.serve.utils.request_cleanup import with_request_cleanup
from lmdeploy.serve.utils.server_utils import validate_json_request


def _create_token_logprobs(token_ids: list[int] | None = None,
                           logprobs: list[dict[int, float]] | None = None):
    """Create raw (logprob, token_id) pairs."""
    if token_ids is None or logprobs is None:
        return None

    token_logprobs = [(tok_logprobs[tok], tok)
                      for tok, tok_logprobs in zip(token_ids, logprobs, strict=True)]
    return token_logprobs or None


def _create_top_logprobs(token_ids: list[int] | None = None,
                         logprobs: list[dict[int, float]] | None = None,
                         top_logprobs_num: int = 0):
    """Create raw top-logprob pairs."""
    if token_ids is None or logprobs is None or top_logprobs_num <= 0:
        return None

    top_logprobs = []
    for tok, tok_logprobs in zip(token_ids, logprobs, strict=True):
        items = [(prob, top_id) for top_id, prob in tok_logprobs.items()]
        if len(items) > top_logprobs_num:
            # Drop the extra selected/target row; expose model top-k only.
            items = [(prob, top_id) for prob, top_id in items if top_id != tok]
        items.sort(key=lambda x: x[0], reverse=True)
        top_logprobs.append(items[:top_logprobs_num])
    return top_logprobs or None


def _create_token_logprobs_from_arrays(token_ids: list[int] | None = None,
                                      logprobs: LogprobArrays | None = None):
    """Create raw (logprob, token_id) pairs from array rows."""
    if token_ids is None or logprobs is None:
        return None
    if len(token_ids) != len(logprobs):
        raise ValueError(f'{len(token_ids)} token ids but {len(logprobs)} logprob rows')
    return list(zip(logprobs.vals[:, 0].tolist(), token_ids)) or None


def _create_flat_top_logprobs(logprobs: LogprobArrays | None = None, top_logprobs_num: int = 0):
    """Encode the top-k columns of array rows as base64 little-endian
    arrays."""
    if logprobs is None or len(logprobs) == 0 or top_logprobs_num <= 0:
        return None
    # Column 0 is the sampled token, already in output_token_logprobs.
    ids = np.ascontiguousarray(logprobs.ids[:, 1:top_logprobs_num + 1], dtype='<i4')
    vals = np.ascontiguousarray(logprobs.vals[:, 1:top_logprobs_num + 1], dtype='<f4')
    return dict(output_top_logprobs_shape=list(ids.shape),
                output_top_logprobs_ids_b64=base64.b64encode(ids.tobytes()).decode('ascii'),
                output_top_logprobs_b64=base64.b64encode(vals.tobytes()).decode('ascii'))


def _check_disaggregation(request: GenerateReqInput, engine_config) -> str:
    """Validate the PD disaggregation fields of a /generate request.

    Prefill side: ``with_cache`` (and usually ``preserve_cache``) returns the
    KV cache handle in the response. Decode side: ``migration_request`` pulls
    that cache; the decode output starts with the prefill's sampled token.
    """
    if not (request.with_cache or request.preserve_cache or request.migration_request is not None):
        return ''
    if not isinstance(engine_config, PytorchEngineConfig):
        return 'with_cache, preserve_cache and migration_request require the PyTorch engine.'
    if request.preserve_cache and not request.with_cache:
        return 'preserve_cache requires with_cache=True.'
    if request.with_cache and request.migration_request is not None:
        return 'with_cache and migration_request cannot be used in the same request.'
    if request.with_cache and request.stream:
        return 'with_cache is not supported with stream=True.'
    if request.logprob_start_len >= 0:
        return 'logprob_start_len is not supported with cache migration.'
    if request.image_data is not None:
        return 'image_data is not supported with cache migration.'
    if request.input_ids is None:
        return 'cache migration requires input_ids.'
    if (request.migration_request is not None and request.return_routed_experts
            and request.routed_experts_start_len != len(request.input_ids)):
        # The decode engine computes positions from len(input_ids) on: the
        # prompt routes come from the prefill response.
        return 'migration_request with return_routed_experts requires routed_experts_start_len=len(input_ids).'
    return ''


def _split_migrated_token(output_ids: list[int], migrated: bool) -> list[int]:
    """Output ids this engine sampled itself.

    A decode request (``migration_request``) first emits the prefill's sampled
    token; it was not sampled here and carries no logprob.
    """
    return output_ids[1:] if migrated and output_ids else output_ids


def check_request(request: GenerateReqInput, server_context) -> str:
    engine_config = server_context.engine_config
    session_manager = server_context.session_manager
    logprobs_mode = getattr(engine_config, 'logprobs_mode', None)
    return_logprob = request.return_logprob
    top_logprobs_num = request.top_logprobs_num or 0
    if top_logprobs_num < 0:
        return f'The top_logprobs_num {request.top_logprobs_num!r} cannot be a negative integer.'
    if top_logprobs_num > 0 and return_logprob is not True:
        return 'top_logprobs_num requires return_logprob=True.'
    if request.top_logprobs_format == 'flat':
        if not isinstance(engine_config, PytorchEngineConfig):
            return "top_logprobs_format='flat' requires the PyTorch engine."
        if request.logprob_start_len >= 0:
            return "top_logprobs_format='flat' does not support logprob_start_len."
    if hasattr(engine_config, 'logprobs_mode') and logprobs_mode is None and return_logprob:
        return f'return_logprob({return_logprob}) requested but not enabled logprobs_mode in engine configuration.'

    if request.return_routed_experts:
        if not hasattr(engine_config, 'enable_return_routed_experts'):
            return f'return_routed_experts is not supported in {type(engine_config).__name__}.'
        if not engine_config.enable_return_routed_experts:
            return ('routed experts requested but not configured in engine configuration. '
                    'May start api_server with --enable-return-routed-experts flag.')

    if (request.prompt is not None) ^ (request.input_ids is None):
        return 'You must specify exactly one of prompt or input_ids'

    if request.prompt is not None and request.prompt == '':
        return 'The prompt must not be an empty string'

    if request.input_ids is not None and len(request.input_ids) == 0:
        return 'The input_ids must not be an empty list'

    if request.max_tokens is not None and request.max_tokens < 0:
        return f'The max_tokens {request.max_tokens!r} must be non-negative.'

    # Check for input-logprob specific requirements.
    if request.logprob_start_len >= 0:
        if not isinstance(engine_config, PytorchEngineConfig) or engine_config.role != EngineRole.Hybrid:
            return 'input logprobs are supported only by the PyTorch hybrid engine.'
        if getattr(getattr(server_context, 'async_engine', None), 'speculative_config', None) is not None:
            return 'logprob_start_len is not supported with speculative decoding.'
        if return_logprob is not True:
            return 'logprob_start_len requires return_logprob=True.'
        if logprobs_mode not in ('raw_logits', 'raw_logprobs'):
            return 'logprob_start_len requires raw_logits or raw_logprobs mode.'
        if request.max_tokens != 0:
            return 'logprob_start_len requires max_tokens=0.'
    elif request.max_tokens is not None and request.max_tokens == 0:
        return f'The max_tokens {request.max_tokens!r} must be a positive integer.'

    pd_error = _check_disaggregation(request, engine_config)
    if pd_error:
        return pd_error

    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

    # check sampling settings
    if request.top_p is not None and not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k is not None and request.top_k < -1:
        return f'The top_k {request.top_k!r} must be -1, 0, or a positive integer.'
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''


def register(router: APIRouter, server_context) -> None:

    @router.post('/generate', dependencies=[Depends(validate_json_request)])
    async def generate(request: GenerateReqInput, raw_request: Request = None):
        error_check_ret = validate_request(request, server_context,
                                           check_request)
        if error_check_ret is not None:
            return error_check_ret

        prompt = request.prompt
        input_ids = request.input_ids
        input_logprobs_requested = request.logprob_start_len >= 0
        flat = request.top_logprobs_format == 'flat'
        meta_cls = GenerateReqFlatMetaOutput if flat else GenerateReqMetaOutput
        image_data = request.image_data
        if image_data is not None:
            # convert to openai format
            image_input = []
            if not isinstance(image_data, list):
                image_data = [image_data]
            for img in image_data:
                if isinstance(img, str):
                    image_input.append(
                        dict(type='image_url', image_url=dict(url=img)))
                else:
                    image_input.append(dict(type='image_url', image_url=img))
            text_input = dict(type='text',
                              text=prompt if prompt else input_ids)
            prompt = [dict(role='user', content=[text_input] + image_input)]
            input_ids = None

        gen_logprobs = (request.top_logprobs_num or 1) if request.return_logprob else None
        migrated = request.migration_request is not None
        gen_config = build_serving_generation_config(
            request,
            server_context,
            logprobs=gen_logprobs,
            logprob_start_len=request.logprob_start_len,
            logprobs_format='array' if flat else 'dict',
            stop_words=request.stop,
            migration_request=request.migration_request,
            with_cache=request.with_cache,
            preserve_cache=request.preserve_cache,
        )

        session = server_context.create_session(request.session_id)
        try:
            prepared_request = await server_context.async_engine.preprocess(
                messages=prompt,
                session_id=session,
                input_ids=input_ids,
                gen_config=gen_config,
                do_preprocess=False,
                media_io_kwargs=request.media_io_kwargs,
                mm_processor_kwargs=request.mm_processor_kwargs)
        except RequestError as e:
            return create_request_error_response(e)
        result_generator = server_context.async_engine.generate(prepared_request, stream_response=True)

        def create_generate_response_json(res,
                                          text,
                                          output_ids,
                                          logprobs,
                                          input_logprobs,
                                          top_logprobs,
                                          input_top_logprobs,
                                          finish_reason,
                                          routed_experts=None,
                                          flat_top_logprobs=None):
            # only output router experts in last chunk
            routed_experts = None if finish_reason is None else routed_experts
            meta = meta_cls(
                finish_reason=dict(
                    type=finish_reason) if finish_reason else None,
                output_token_logprobs=logprobs,
                input_token_logprobs=input_logprobs,
                output_top_logprobs=top_logprobs,
                input_top_logprobs=input_top_logprobs,
                prompt_tokens=res.input_token_len,
                routed_experts=routed_experts,
                completion_tokens=res.generate_token_len,
                **(flat_top_logprobs or {}))

            response = GenerateReqOutput(text=text,
                                         output_ids=output_ids,
                                         meta_info=meta)
            return response.model_dump_json()

        async def generate_stream_generator():
            migrated_pending = migrated
            try:
                async for res in result_generator:
                    text = res.response or ''
                    output_ids = res.token_ids
                    routed_experts = res.routed_experts
                    flat_top_logprobs = None
                    if input_logprobs_requested:
                        input_logprob_token_ids = res.logprob_token_ids
                        logprobs = None
                        input_logprobs = _create_token_logprobs(input_logprob_token_ids, res.logprobs)
                        top_logprobs = None
                        input_top_logprobs = (_create_top_logprobs(input_logprob_token_ids, res.logprobs,
                                                                   request.top_logprobs_num)
                                              if request.top_logprobs_num else None)
                    else:
                        sampled_ids = res.token_ids
                        if migrated_pending and sampled_ids:
                            # The migrated token may share a chunk with the first local sample.
                            sampled_ids, migrated_pending = sampled_ids[1:], False
                        if flat:
                            logprobs = _create_token_logprobs_from_arrays(sampled_ids, res.logprobs)
                            top_logprobs = None
                            flat_top_logprobs = _create_flat_top_logprobs(res.logprobs, request.top_logprobs_num or 0)
                        else:
                            logprobs = _create_token_logprobs(sampled_ids, res.logprobs)
                            top_logprobs = (_create_top_logprobs(sampled_ids, res.logprobs, request.top_logprobs_num)
                                            if request.top_logprobs_num else None)
                        input_logprobs = None
                        input_top_logprobs = None
                    response_json = create_generate_response_json(
                        res,
                        text,
                        output_ids,
                        logprobs,
                        input_logprobs,
                        top_logprobs,
                        input_top_logprobs,
                        res.finish_reason,
                        routed_experts=routed_experts,
                        flat_top_logprobs=flat_top_logprobs)
                    yield f'data: {response_json}\n\n'
            except RequestError as e:
                yield f'data: {json.dumps({"error": request_error_payload(e)}, ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'

        if request.stream:
            stream_generator = with_request_cleanup(
                generate_stream_generator(), [result_generator], [session],
                server_context.session_manager)
            return StreamingResponse(stream_generator,
                                     media_type='text/event-stream')

        async def _inner_call():
            text = ''
            output_ids = []
            logprobs = []
            input_logprobs = []
            input_token_logprobs = None
            input_top_logprobs = None
            input_logprob_token_ids = None
            cache_block_ids = None
            try:
                async with aclosing(
                        with_request_cleanup(
                            result_generator, [result_generator], [session],
                            server_context.session_manager)) as generator:
                    async for res in generator:
                        if await raw_request.is_disconnected():
                            # Abort the request if the client disconnects.
                            await session.async_abort()
                            return create_error_response(HTTPStatus.BAD_REQUEST,
                                                         'Client disconnected')
                        text += res.response or ''
                        output_ids.extend(res.token_ids or [])
                        if request.with_cache and res.cache_block_ids is not None:
                            cache_block_ids = res.cache_block_ids
                        if input_logprobs_requested:
                            input_logprob_token_ids = res.logprob_token_ids
                            if res.logprobs:
                                input_logprobs = res.logprobs
                        elif res.logprobs:
                            if flat:
                                logprobs.append(res.logprobs)
                            else:
                                logprobs.extend(res.logprobs)
            except RequestError as e:
                return create_request_error_response(e)

            sampled_ids = _split_migrated_token(output_ids, migrated)
            flat_top_logprobs = None
            if flat:
                rows = LogprobArrays.concat(logprobs) if logprobs else None
                output_token_logprobs = _create_token_logprobs_from_arrays(sampled_ids, rows)
                output_top_logprobs = None
                flat_top_logprobs = _create_flat_top_logprobs(rows, request.top_logprobs_num or 0)
            else:
                output_token_logprobs = _create_token_logprobs(sampled_ids, logprobs) if logprobs else None
                output_top_logprobs = (_create_top_logprobs(sampled_ids, logprobs, request.top_logprobs_num)
                                       if request.top_logprobs_num and logprobs else None)
            if input_logprobs_requested and input_logprobs:
                input_token_logprobs = _create_token_logprobs(input_logprob_token_ids, input_logprobs)
                input_top_logprobs = (_create_top_logprobs(input_logprob_token_ids, input_logprobs,
                                                           request.top_logprobs_num)
                                      if request.top_logprobs_num else None)

            meta = meta_cls(
                finish_reason=dict(
                    type=res.finish_reason) if res.finish_reason else None,
                output_token_logprobs=output_token_logprobs or None,
                input_token_logprobs=input_token_logprobs,
                output_top_logprobs=output_top_logprobs,
                input_top_logprobs=input_top_logprobs,
                prompt_tokens=res.input_token_len,
                routed_experts=res.routed_experts,
                completion_tokens=res.generate_token_len,
                **(flat_top_logprobs or {}))
            response = GenerateReqOutput(text=text,
                                         output_ids=output_ids,
                                         meta_info=meta)
            if request.with_cache:
                # Cache ownership uses the engine session, not the request's session_id.
                response.cache_session_id = session.session_id
                response.cache_block_ids = cache_block_ids or []
                response.remote_token_ids = output_ids[-1:]
            return response

        return await _inner_call()
