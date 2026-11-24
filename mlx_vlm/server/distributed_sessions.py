# Copyright © 2026 Neywa Labs Inc.
"""Control the existing continuation sessions; no separate token/checkpoint store."""

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import continuation
from .runtime import runtime

SESSION_ROUTE = "/internal/distributed/session"


class SessionAction(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    action: Literal["pause", "dispose", "status"]


def router(authenticate):
    api = APIRouter(dependencies=[Depends(authenticate)], tags=["Distributed sessions"])

    @api.post(SESSION_ROUTE)
    async def control(body: SessionAction):
        if runtime.distributed_worker is None:
            raise HTTPException(409, "This is not a distributed worker")
        deadline = time.monotonic() + 60
        while True:
            with continuation.CHECKPOINTS.lock:
                record = continuation.CHECKPOINTS.records.get(body.id)
            if record is None:
                return {"state": "absent"}
            session = record.session
            if session is not None:
                break
            if body.action == "status":
                return {"state": "preparing"}
            if time.monotonic() >= deadline:
                raise HTTPException(
                    504, "Session preparation did not reach a safe control point"
                )
            await asyncio.sleep(0.01)
        if body.action == "pause":
            if record.lease != session.lease:
                return {"state": "superseded"}
            continuation.CHECKPOINTS.release(record)
        elif body.action == "dispose":
            session.dispose("Distributed continuation cancelled or expired.")
        else:
            return {"state": session.distributed_ack, "finished": session.finished}
        while not session.finished and session.distributed_ack not in (
            {"paused", "disposed"} if body.action == "pause" else {"disposed"}
        ):
            if time.monotonic() >= deadline:
                raise HTTPException(
                    504, "Distributed session transition did not reach all ranks"
                )
            await asyncio.sleep(0.01)
        if body.action == "dispose":
            with continuation.CHECKPOINTS.lock:
                continuation.CHECKPOINTS._remove_locked(
                    [
                        key
                        for key, value in continuation.CHECKPOINTS.records.items()
                        if value.session is session
                    ]
                )
        return {"state": "finished" if session.finished else session.distributed_ack}

    return api
