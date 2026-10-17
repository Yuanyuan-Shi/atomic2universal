"""Closed-loop policy interface backed by the official GR00T N1.5 inference server.

The checkpoint runs in the `a2u-policy` env behind the RoboCasa365 fork's
`scripts/inference_service.py`, which applies the official `panda_omron` transforms (image
crop/resize, state and action normalization) around `Gr00tPolicy`. This module talks to it with
the fork's own `ExternalRobotInferenceClient`, packs one observation in the request layout of the
official eval (time and batch axes on video/state, the instruction as a one-element array),
validates the returned action chunk, and flattens it into the 12-D row layout that
`robocasa.utils.env_utils.convert_action` maps back to a gym action. The official client sets no
socket timeout, so a subclass applies one: a dead or stuck server fails loudly instead of hanging.
"""

from typing import Any

import numpy as np
import zmq
from gr00t.eval.service import ExternalRobotInferenceClient

from evaluation.config import EvalSpec

# Action keys returned by the server and their widths, in the order of
# robocasa.utils.env_utils.convert_action (rows 0:3, 3:6, 6:7, 7:11, 11:12).
ACTION_LAYOUT = (
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
    ("action.base_motion", 4),
    ("action.control_mode", 1),
)
LANGUAGE_KEY = "annotation.human.task_description"


class PolicyError(RuntimeError):
    """The policy server is unreachable, timed out, or returned an unexpected action."""


class TimeoutInferenceClient(ExternalRobotInferenceClient):
    """Official ZMQ client whose socket honors `timeout_ms` (the base class stores but never applies it)."""

    def _init_socket(self) -> None:
        previous_socket = getattr(self, "socket", None)
        if previous_socket is not None:  # ping() re-creates the socket after a failure
            previous_socket.close(linger=0)
        super()._init_socket()
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)


class Policy:
    """GR00T N1.5 policy served over ZMQ; returns one action chunk per query."""

    def __init__(self, client: TimeoutInferenceClient, action_horizon: int) -> None:
        self.client = client
        self.action_horizon = action_horizon
        self.address = f"tcp://{client.host}:{client.port}"
        self.timeout_s = client.timeout_ms / 1000

    def _server_hint(self) -> str:
        return (
            f"Check that the policy server at {self.address} is running "
            "(`ps aux | grep inference_service`) and read its log; start one with "
            "`bash evaluation/serve_policy.sh --port N` or pass --launch-server."
        )

    def reset(self) -> None:
        """Start a new episode. GR00T N1.5 keeps no state between queries, so this only
        confirms that the server answers before the episode begins."""
        if not self.client.ping():
            raise PolicyError(f"Policy server did not answer a ping within {self.timeout_s:.0f} s. {self._server_hint()}")

    def act(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Query the server with one env observation and return actions of shape (action_horizon, 12)."""
        request = {
            key: np.asarray(value)[None, None]
            for key, value in observation.items()
            if key.startswith("video.") or key.startswith("state.")
        }
        if not request:
            raise PolicyError(f"Observation has no video.* or state.* keys: {sorted(observation)}")
        request[LANGUAGE_KEY] = np.array([instruction])
        try:
            response = self.client.get_action(request)
        except zmq.error.Again as error:
            raise PolicyError(
                f"Policy server did not answer get_action within {self.timeout_s:.0f} s. {self._server_hint()}"
            ) from error
        except RuntimeError as error:
            raise PolicyError(f"Policy server returned an error: {error}. {self._server_hint()}") from error
        return self._flatten_chunk(response)

    def _flatten_chunk(self, response: Any) -> np.ndarray:
        expected_keys = [key for key, _ in ACTION_LAYOUT]
        if not isinstance(response, dict) or sorted(response) != sorted(expected_keys):
            found = sorted(response) if isinstance(response, dict) else type(response).__name__
            raise PolicyError(
                f"Server returned action keys {found}, expected {expected_keys}. "
                "Is it serving the panda_omron data config with embodiment new_embodiment?"
            )
        parts = []
        for key, width in ACTION_LAYOUT:
            value = np.asarray(response[key])
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
                raise PolicyError(f"{key} has shape {value.shape}, expected (1, chunk_length, {width})")
            if value.shape[1] < self.action_horizon:
                raise PolicyError(
                    f"{key} has {value.shape[1]} steps, fewer than action_horizon {self.action_horizon} "
                    "in the eval config"
                )
            parts.append(value[0, : self.action_horizon])
        chunk = np.concatenate(parts, axis=-1)
        if not np.all(np.isfinite(chunk)):
            raise PolicyError("Server returned non-finite actions")
        return chunk


def load_policy(eval_spec: EvalSpec, host: str = "localhost", port: int = 5555, timeout_s: float = 60.0) -> Policy:
    """Connect to a running policy server, check that it answers, and return the Policy.

    The server must already have loaded `eval_spec.checkpoint` (see evaluation/serve_policy.sh);
    the action horizon is taken from the eval config, as in the official run_eval.py.
    """
    client = TimeoutInferenceClient(host=host, port=port, timeout_ms=int(timeout_s * 1000))
    policy = Policy(client, eval_spec.action_horizon)
    policy.reset()
    return policy
