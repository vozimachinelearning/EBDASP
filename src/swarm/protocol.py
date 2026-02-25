import json
import zlib
from typing import Any, Dict

from .messages import Message, message_from_dict


class Protocol:
    def __init__(self, version: str = "0.1.0") -> None:
        self.version = version

    def encode(self, message: Message) -> str:
        payload: Dict[str, Any] = message.to_dict()
        payload["protocol_version"] = self.version
        return json.dumps(payload, ensure_ascii=False)

    def decode(self, payload: str) -> Message:
        data = json.loads(payload)
        return message_from_dict(data)

    def encode_compact(self, message: Message) -> bytes:
        payload: Dict[str, Any] = message.to_dict()
        payload["protocol_version"] = self.version
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def decode_compact(self, payload: bytes) -> Message:
        data = json.loads(payload.decode("utf-8"))
        return message_from_dict(data)

    def encode_binary(self, message: Message) -> bytes:
        return zlib.compress(self.encode_compact(message), level=9)

    def decode_binary(self, payload: bytes) -> Message:
        data = zlib.decompress(payload)
        return self.decode_compact(data)
