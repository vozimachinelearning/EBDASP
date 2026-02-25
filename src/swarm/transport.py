from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import RNS

from .messages import AnnounceCapabilities, Heartbeat, QueryRequest, QueryResponse, RouteRequest, RouteResponse
from .protocol import Protocol
class Transport:
    def __init__(
        self,
        node_id: str,
        heartbeat_ttl_seconds: float = 15.0,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self.node_id = node_id
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
        self._time_provider = time_provider or time.time
        self._workers: Dict[str, Worker] = {}
        self._announcements: Dict[str, AnnounceCapabilities] = {}
        self._last_seen: Dict[str, float] = {}
        self._activity: List[Dict[str, Any]] = []
        self._activity_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register_worker(self, node_id: str, worker: Worker) -> None:
        self._workers[node_id] = worker

    def announce(self, announcement: AnnounceCapabilities) -> None:
        self._announcements[announcement.node_id] = announcement
        self._last_seen[announcement.node_id] = self._time_provider()
        self.emit_activity(
            "announce",
            node_id=announcement.node_id,
            payload={
                "domains": list(announcement.domains),
                "collections": list(announcement.collections),
            },
        )

    def heartbeat(self, node_id: str) -> None:
        self._last_seen[node_id] = self._time_provider()
        self.emit_activity("heartbeat", node_id=node_id, payload={})

    def available_nodes(self, domain: Optional[str] = None) -> List[str]:
        self._prune_stale()
        candidates = list(self._announcements.values())
        if domain:
            candidates = [item for item in candidates if domain in item.domains]
        return [item.node_id for item in candidates]

    def live_status(self) -> List[Dict[str, Any]]:
        self._prune_stale()
        now = self._time_provider()
        results: List[Dict[str, Any]] = []
        for node_id, announcement in self._announcements.items():
            last_seen = self._last_seen.get(node_id, 0.0)
            results.append(
                {
                    "node_id": node_id,
                    "domains": list(announcement.domains),
                    "collections": list(announcement.collections),
                    "last_seen_seconds": max(0.0, now - last_seen),
                }
            )
        return results

    def subscribe_activity(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._activity_callbacks.append(callback)

    def get_activity(self, since_index: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        events = self._activity[since_index:]
        return events, len(self._activity)

    def emit_activity(
        self,
        event: str,
        node_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        activity = {
            "timestamp": self._time_provider(),
            "event": event,
            "node_id": node_id,
            "payload": payload or {},
        }
        self._activity.append(activity)
        for callback in self._activity_callbacks:
            callback(activity)
        return activity

    def route(self, request: RouteRequest) -> RouteResponse:
        candidates = self.available_nodes(domain=request.domain)
        node_ids = candidates[: request.limit]
        self.emit_activity(
            "route",
            node_id=None,
            payload={
                "query_id": request.query_id,
                "domain": request.domain,
                "candidates": list(candidates),
            },
        )
        return RouteResponse(query_id=request.query_id, node_ids=node_ids)

    def send_query(self, node_id: str, request: QueryRequest) -> QueryResponse:
        worker = self._workers.get(node_id)
        if worker is None:
            raise ValueError(f"Unknown worker node: {node_id}")
        self.emit_activity(
            "query_sent",
            node_id=node_id,
            payload={
                "query_id": request.query_id,
                "question": request.question,
                "domain": request.domain,
                "recursion_budget": request.recursion_budget,
            },
        )
        response = worker.handle_query(request)
        self.emit_activity(
            "query_response",
            node_id=node_id,
            payload={
                "query_id": response.query_id,
                "confidence": response.confidence,
                "claims_count": len(response.claims),
                "evidence_count": len(response.evidence),
                "next_queries": len(response.next_queries),
            },
        )
        return response

    def _prune_stale(self) -> None:
        now = self._time_provider()
        expired: List[str] = []
        for node_id, last_seen in self._last_seen.items():
            if now - last_seen > self.heartbeat_ttl_seconds:
                expired.append(node_id)
        for node_id in expired:
            self._last_seen.pop(node_id, None)
            self._announcements.pop(node_id, None)


class NetworkTransport(Transport):
    def __init__(
        self,
        node_id: str,
        protocol: Protocol,
        rns_config_dir: Optional[str] = None,
        app_name: str = "swarm",
        aspect: str = "transport",
        heartbeat_ttl_seconds: float = 15.0,
        response_timeout_seconds: float = 15.0,
        announce_interval_seconds: float = 10.0,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(node_id=node_id, heartbeat_ttl_seconds=heartbeat_ttl_seconds, time_provider=time_provider)
        self.protocol = protocol
        self._app_name = app_name
        self._aspect = aspect
        self._response_timeout_seconds = response_timeout_seconds
        self._announce_interval_seconds = announce_interval_seconds
        self._node_destinations: Dict[str, str] = {}
        self._pending_responses: Dict[str, QueryResponse] = {}
        self._pending_events: Dict[str, threading.Event] = {}
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        if rns_config_dir:
            RNS.Reticulum(rns_config_dir)
        else:
            RNS.Reticulum()
        self._identity = RNS.Identity()
        self._destination = RNS.Destination(
            self._identity,
            RNS.Destination.IN,
            RNS.Destination.SINGLE,
            self._app_name,
            self._aspect,
        )
        self._destination.set_packet_callback(self._on_packet)
        self._announce_handler = _AnnounceHandler(self)
        RNS.Transport.register_announce_handler(self._announce_handler)
        self._destination_hash = self._destination.hash
        self._destination_hash_hex = self._destination_hash.hex()
        threading.Thread(target=self._announce_loop, daemon=True).start()

    def announce(self, announcement: AnnounceCapabilities) -> None:
        if not announcement.destination_hash:
            announcement = AnnounceCapabilities(
                node_id=announcement.node_id,
                domains=list(announcement.domains),
                collections=list(announcement.collections),
                timestamp=announcement.timestamp,
                signature=announcement.signature,
                destination_hash=self._destination_hash_hex,
            )
        super().announce(announcement)
        if announcement.node_id in self._workers:
            payload = self.protocol.encode_compact(announcement)
            self._destination.announce(app_data=payload)

    def heartbeat(self, node_id: str) -> None:
        super().heartbeat(node_id)
        heartbeat = Heartbeat(
            node_id=node_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._time_provider())),
            status="ok",
        )
        payload = self.protocol.encode_compact(heartbeat)
        for destination_hash in list(self._node_destinations.values()):
            self._send_packet(destination_hash, payload)

    def send_query(self, node_id: str, request: QueryRequest) -> QueryResponse:
        worker = self._workers.get(node_id)
        if worker is not None:
            return super().send_query(node_id, request)
        destination_hash = self._node_destinations.get(node_id)
        if destination_hash is None:
            raise ValueError(f"Unknown remote node: {node_id}")
        constraints = dict(request.constraints)
        constraints["target_node_id"] = node_id
        constraints["reply_to"] = self._destination_hash_hex
        network_request = QueryRequest(
            query_id=request.query_id,
            question=request.question,
            domain=request.domain,
            recursion_budget=request.recursion_budget,
            constraints=constraints,
        )
        payload = self.protocol.encode_compact(network_request)
        event = threading.Event()
        with self._pending_lock:
            self._pending_events[request.query_id] = event
        self._send_packet(destination_hash, payload)
        if not event.wait(self._response_timeout_seconds):
            with self._pending_lock:
                self._pending_events.pop(request.query_id, None)
                self._pending_responses.pop(request.query_id, None)
            raise TimeoutError(f"No response for query {request.query_id}")
        with self._pending_lock:
            response = self._pending_responses.pop(request.query_id, None)
            self._pending_events.pop(request.query_id, None)
        if response is None:
            raise TimeoutError(f"No response for query {request.query_id}")
        return response

    def _announce_loop(self) -> None:
        while not self._stop_event.is_set():
            for announcement in list(self._announcements.values()):
                if announcement.node_id in self._workers:
                    self.announce(announcement)
            time.sleep(self._announce_interval_seconds)

    def _on_packet(self, packet: RNS.Packet) -> None:
        try:
            message = self.protocol.decode_compact(packet.data)
        except Exception:
            return
        if isinstance(message, AnnounceCapabilities):
            if not message.destination_hash:
                message.destination_hash = packet.destination_hash.hex()
            if message.destination_hash:
                self._node_destinations[message.node_id] = message.destination_hash
            super().announce(message)
            return
        if isinstance(message, Heartbeat):
            self._last_seen[message.node_id] = self._time_provider()
            self.emit_activity("heartbeat", node_id=message.node_id, payload={})
            return
        if isinstance(message, QueryRequest):
            target = message.constraints.get("target_node_id")
            if target and target not in self._workers:
                return
            worker = self._workers.get(target) if target else None
            if worker is None and self._workers:
                worker = next(iter(self._workers.values()))
            if worker is None:
                return
            response = worker.handle_query(message)
            reply_to = message.constraints.get("reply_to")
            if reply_to:
                payload = self.protocol.encode_compact(response)
                self._send_packet(reply_to, payload)
            return
        if isinstance(message, QueryResponse):
            with self._pending_lock:
                event = self._pending_events.get(message.query_id)
                if event is None:
                    return
                self._pending_responses[message.query_id] = message
                event.set()
            return

    def _send_packet(self, destination_hash: str, payload: bytes) -> None:
        destination = RNS.Destination(
            None,
            RNS.Destination.OUT,
            RNS.Destination.SINGLE,
            self._app_name,
            self._aspect,
            bytes.fromhex(destination_hash),
        )
        packet = RNS.Packet(destination, payload)
        packet.send()


class _AnnounceHandler:
    def __init__(self, transport: NetworkTransport) -> None:
        self.transport = transport

    def received_announce(self, destination_hash: bytes, announced_identity: Any, app_data: Optional[bytes]) -> None:
        if not app_data:
            return
        try:
            message = self.transport.protocol.decode_compact(app_data)
        except Exception:
            return
        if isinstance(message, AnnounceCapabilities):
            if not message.destination_hash:
                message.destination_hash = destination_hash.hex()
            if message.destination_hash:
                self.transport._node_destinations[message.node_id] = message.destination_hash
            super(NetworkTransport, self.transport).announce(message)
