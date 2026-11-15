# Copyright © 2026 Neywa Labs Inc.
"""Public, application-independent cluster management contract."""

import re
from ipaddress import IPv4Address
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


def worker_origin(value):
    url = urlsplit(value)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        raise ValueError(
            "Worker endpoint must be an HTTP(S) origin without credentials"
        )
    return value.rstrip("/")


def model_directory(value):
    # rsync's remote argument is interpreted by a remote shell as well.
    if not re.fullmatch(r"/[\w/.-]+", value):
        raise ValueError("model_dir must be absolute without shell metacharacters")
    return value


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClusterError(Contract):
    code: str
    message: str
    retryable: bool = False
    action: str | None = None


class ClusterErrorResponse(Contract):
    detail: ClusterError


class RDMAPair(Contract):
    """Explicit direct link from this coordinator to one approved worker."""

    local_address: str
    local_device: str = Field(pattern=r"^rdma_[\w]+$")
    remote_device: str = Field(pattern=r"^rdma_[\w]+$")

    @field_validator("local_address")
    @classmethod
    def address(cls, value):
        return str(IPv4Address(value))


class SSHTransport(Contract):
    """Optional preconfigured transfer access, never SSH enrollment."""

    destination: str = Field(pattern=r"^[\w][\w.@-]*$")
    python: str = Field(default="python3", min_length=1)
    rdma: RDMAPair | None = None


class Memory(Contract):
    total_bytes: int | None = None
    used_bytes: int | None = None
    mlx_active_bytes: int | None = None
    mlx_cache_bytes: int | None = None
    mlx_peak_bytes: int | None = None


class NodeIdentity(Contract):
    id: str
    name: str
    physical_id: str
    os: str
    mlx_version: str
    mlx_vlm_version: str
    memory: Memory
    measured_at: float
    backends: list[str] = Field(default_factory=list)
    worker: dict | None = None
    model_dir: str | None = None
    # Negotiated link properties do not prove usable RDMA collectives.
    links: list[dict] = Field(default_factory=list)
    rdma_validated: bool = False
    checkpoint_transfer: list[str] = Field(default_factory=lambda: ["ssh_rsync"])


class RegisterNode(Contract):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    endpoint: str
    api_key: SecretStr
    name: str | None = None
    ssh: SSHTransport | None = None

    @field_validator("endpoint")
    @classmethod
    def origin(cls, value):
        return worker_origin(value)


class UpdateNode(Contract):
    endpoint: str | None = None
    api_key: SecretStr | None = None
    name: str | None = None
    ssh: SSHTransport | None = None

    @field_validator("endpoint")
    @classmethod
    def origin(cls, value):
        return worker_origin(value) if value is not None else None


class Node(Contract):
    id: str
    name: str
    endpoint: str
    ssh: SSHTransport | None = None
    identity: NodeIdentity
    reachable: bool = True
    stale: bool = False
    checked_at: float
    error: ClusterError | None = None


class Members(Contract):
    node_ids: list[str] = Field(min_length=2)
    backend: Literal["jaccl", "ring", "mpi", "nccl"] = "jaccl"
    model: str | None = None
    revision: str | None = None

    @field_validator("node_ids")
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Selected node IDs must be unique")
        return value


class StartGroup(Contract):
    mode: Literal["attach", "launch"] = "attach"
    model: str | None = None
    revision: str | None = None
    fast_sync: bool | None = Field(
        default=None,
        strict=True,
        description="Experimental fast synchronization for this managed launch; "
        "omit to use launch.fast_sync from the server configuration.",
    )

    @model_validator(mode="after")
    def fast_sync_requires_launch(self):
        if self.fast_sync is not None and self.mode != "launch":
            raise ValueError("fast_sync requires mode='launch' and fresh workers")
        return self


class StopGroup(Contract):
    drain_timeout_seconds: float = Field(default=60, gt=0, le=7200)
    discard_continuations: bool = False


class PrepareModel(Contract):
    model: str = Field(min_length=1)
    revision: str | None = None
    node_ids: list[str] | None = Field(default=None, min_length=1)
    transport: Literal["auto", "jaccl_rdma", "ssh_rsync"] = "ssh_rsync"

    @field_validator("node_ids")
    @classmethod
    def unique(cls, value):
        if value is not None and len(value) != len(set(value)):
            raise ValueError("Target node IDs must be unique")
        return value


class Operation(Contract):
    id: str
    kind: str
    status: Literal["running", "succeeded", "failed"] = "running"
    phase: str = "accepted"
    created_at: float
    updated_at: float
    result: dict | None = None
    error: ClusterError | None = None


class ClusterEvent(Contract):
    id: int
    type: str
    timestamp: float
    data: dict


class Capabilities(Contract):
    streaming: bool = True
    continuous_batching: bool = True
    resumable: bool = True
    prompt_progress: bool = True
    generation_progress: bool = True
    stored_responses: bool = False
    previous_response_id: bool = False
    authenticated_weight_delivery: bool = True
    checkpoint_transfer: list[str] = Field(
        default_factory=lambda: ["ssh_rsync", "jaccl_rdma"]
    )


class AggregateMemory(Contract):
    total_bytes: int
    used_bytes: int
    complete: bool
    stale_node_ids: list[str]
    missing_node_ids: list[str]
    physical_devices: int


class ClusterState(Contract):
    state: Literal["stopped", "starting", "ready", "draining", "failed"]
    coordinator: NodeIdentity
    inference_coordinator_id: str | None
    selected_members: list[str]
    running_members: list[str]
    backend: str | None
    active_model: str | None
    active_requests: int
    ownership: Literal["external", "server"]
    fast_sync: bool | None = None
    memory: AggregateMemory
    capabilities: Capabilities = Field(default_factory=Capabilities)
    operation_id: str | None = None


class PromptProgress(Contract):
    type: Literal["response.prompt_progress"] = "response.prompt_progress"
    response_id: str
    status: Literal["started", "in_progress", "completed"]
    processed_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class GenerationProgress(Contract):
    type: Literal["response.generation_progress"] = "response.generation_progress"
    response_id: str
    output_tokens: int = Field(ge=0)
    timings: dict[str, float] | None = None
