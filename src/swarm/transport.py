from __future__ import annotations

import threading
import json
import uuid
import time
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import RNS

from .messages import (
    AnnounceCapabilities,
    ContextUpdate,
    Heartbeat,
    QueryRequest,
    QueryResponse,
    RouteRequest,
    RouteResponse,
    TextMessage,
    TaskAssignment,
    TaskResult,
)
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
        self._completion_ledger: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._context_chunk_buffer: Dict[str, Dict[str, Any]] = {}
        self._max_context_chunk_size = 8000

    def register_worker(self, node_id: str, worker: Worker) -> None:
        with self._lock:
            self._workers[node_id] = worker

    def announce(self, announcement: AnnounceCapabilities) -> None:
        with self._lock:
            if announcement.node_id not in self._announcements:
                print(f"[Transport] Node discovered: {announcement.node_id}")
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
        with self._lock:
            self._last_seen[node_id] = self._time_provider()
        self.emit_activity("heartbeat", node_id=node_id, payload={})

    def available_nodes(self, domain: Optional[str] = None) -> List[str]:
        self._prune_stale()
        with self._lock:
            candidates = list(self._announcements.values())
        if domain:
            candidates = [item for item in candidates if domain in item.domains]
        return [item.node_id for item in candidates]

    def live_status(self) -> List[Dict[str, Any]]:
        self._prune_stale()
        now = self._time_provider()
        results: List[Dict[str, Any]] = []
        with self._lock:
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
            try:
                callback(activity)
            except Exception:
                continue
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

    def send_task(self, node_id: str, assignment: TaskAssignment) -> TaskResult:
        print(f"[Transport] Sending Task {assignment.task.task_id} to {node_id}")
        worker = self._workers.get(node_id)
        if worker is not None:
            return worker.handle_task(assignment)
        raise NotImplementedError("Remote task sending not implemented in base Transport")

    def send_message(self, node_id: str, text: str, sender: Optional[str] = None) -> bool:
        message = TextMessage(
            message_id=str(uuid.uuid4()),
            sender=sender or self.node_id,
            recipient=node_id,
            text=text,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._time_provider())),
        )
        self.emit_activity(
            "message_sent",
            node_id=node_id,
            payload={
                "message_id": message.message_id,
                "sender": message.sender,
                "recipient": message.recipient,
                "text": message.text,
            },
        )
        if node_id in self._workers:
            self.emit_activity(
                "message_received",
                node_id=node_id,
                payload={
                    "message_id": message.message_id,
                    "sender": message.sender,
                    "recipient": message.recipient,
                    "text": message.text,
                },
            )
            return True
        return False

    def send_context_update(self, node_id: str, content: str, context_id: Optional[str] = None) -> bool:
        if node_id not in self._workers:
            return False
        context_id_value = context_id or str(uuid.uuid4())
        chunks = self._build_context_chunks(content, context_id_value)
        for chunk in chunks:
            self._handle_context_update(chunk, self.node_id, context_id_value)
        return True

    def get_completion_ledger(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._completion_ledger)

    def record_completion(self, payload: Dict[str, Any]) -> None:
        key = payload.get("assignment_id") or payload.get("result_id") or payload.get("task_id")
        if not key:
            return
        with self._lock:
            self._completion_ledger[key] = payload

    def _record_completion_from_content(self, content: str, source_node: str, context_id: str) -> None:
        try:
            payload = json.loads(content)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "task_completion":
            return
        self.record_completion(payload)
        self.emit_activity(
            "completion_ledger_update",
            node_id=source_node,
            payload={
                "task_id": payload.get("task_id"),
                "assignment_id": payload.get("assignment_id"),
                "result_id": payload.get("result_id"),
                "context_id": context_id,
            },
        )

    def _extract_global_context_payload(self, content: str) -> Optional[Dict[str, Any]]:
        lines = content.splitlines()
        if not lines:
            return None
        if lines[0].strip() != "GLOBAL_CONTEXT":
            return None
        context_id_value: Optional[str] = None
        version_value: Optional[int] = None
        text_lines: List[str] = []
        in_text = False
        for line in lines[1:]:
            if line.startswith("context_id:"):
                context_id_value = line.split(":", 1)[1].strip()
                continue
            if line.startswith("version:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    version_value = int(raw)
                except Exception:
                    version_value = None
                continue
            if line.startswith("text:"):
                in_text = True
                continue
            if in_text:
                text_lines.append(line)
        if not context_id_value:
            return None
        return {
            "context_id": context_id_value,
            "version": version_value,
            "text": "\n".join(text_lines).strip(),
        }

    def _build_context_chunks(self, content: str, context_id: str) -> List[str]:
        if len(content) <= self._max_context_chunk_size:
            return [content]
        payloads = [
            content[i : i + self._max_context_chunk_size]
            for i in range(0, len(content), self._max_context_chunk_size)
        ]
        total = len(payloads)
        chunks = []
        for index, payload in enumerate(payloads):
            chunks.append(
                "\n".join(
                    [
                        "CONTEXT_CHUNK",
                        f"context_id: {context_id}",
                        f"chunk_index: {index}",
                        f"chunk_total: {total}",
                        "payload:",
                        payload,
                    ]
                )
            )
        return chunks

    def _parse_context_chunk(self, content: str) -> Optional[Dict[str, Any]]:
        lines = content.splitlines()
        if not lines:
            return None
        if lines[0].strip() != "CONTEXT_CHUNK":
            return None
        chunk_index = None
        chunk_total = None
        context_id_value = None
        payload_lines: List[str] = []
        in_payload = False
        for line in lines[1:]:
            if line.startswith("context_id:"):
                context_id_value = line.split(":", 1)[1].strip()
                continue
            if line.startswith("chunk_index:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    chunk_index = int(raw)
                except Exception:
                    chunk_index = None
                continue
            if line.startswith("chunk_total:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    chunk_total = int(raw)
                except Exception:
                    chunk_total = None
                continue
            if line.startswith("payload:"):
                in_payload = True
                continue
            if in_payload:
                payload_lines.append(line)
        if context_id_value is None or chunk_index is None or chunk_total is None:
            return None
        return {
            "context_id": context_id_value,
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "payload": "\n".join(payload_lines),
        }

    def _handle_context_update(self, content: str, source_node: str, context_id: str) -> None:
        chunk = self._parse_context_chunk(content)
        if not chunk:
            self._handle_full_context_update(content, source_node, context_id)
            return
        chunk_id = chunk["context_id"]
        with self._lock:
            entry = self._context_chunk_buffer.get(chunk_id)
            if not entry:
                entry = {
                    "total": chunk["chunk_total"],
                    "received": {},
                    "source_node": source_node,
                    "timestamp": self._time_provider(),
                }
                self._context_chunk_buffer[chunk_id] = entry
            entry["received"][chunk["chunk_index"]] = chunk["payload"]
            if len(entry["received"]) < entry["total"]:
                return
            payloads = [entry["received"].get(i, "") for i in range(entry["total"])]
            full_content = "".join(payloads)
            self._context_chunk_buffer.pop(chunk_id, None)
        self._handle_full_context_update(full_content, source_node, context_id)

    def _handle_full_context_update(self, content: str, source_node: str, context_id: str) -> None:
        for worker in self._workers.values():
            if worker.store:
                tags = ["context_update"]
                payload = self._extract_global_context_payload(content)
                if payload:
                    context_id_value = payload.get("context_id")
                    if context_id_value:
                        tags.append("global_context")
                        tags.append(f"context_id:{context_id_value}")
                    version_value = payload.get("version")
                    if version_value is not None:
                        tags.append(f"context_version:{version_value}")
                worker.store.add_memory(content, source_node, tags=tags)
        self._record_completion_from_content(content, source_node, context_id)
        self.emit_activity(
            "context_update",
            node_id=source_node,
            payload={"context_id": context_id},
        )

    def _prune_stale(self) -> None:
        now = self._time_provider()
        expired: List[str] = []
        with self._lock:
            for node_id, last_seen in self._last_seen.items():
                # If we have an active link, don't prune even if heartbeat is old
                # Need to access _links safely as it's defined in NetworkTransport subclass
                if getattr(self, '_links', None) and node_id in self._links:
                     if self._links[node_id].status == RNS.Link.ACTIVE:
                         continue

                if now - last_seen > self.heartbeat_ttl_seconds:
                    expired.append(node_id)
            for node_id in expired:
                print(f"[Transport] Node lost (stale): {node_id}")
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
        heartbeat_ttl_seconds: float = 120.0,
        response_timeout_seconds: float = 15.0,
        announce_interval_seconds: float = 30.0,
        identity_path: Optional[str] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(node_id=node_id, heartbeat_ttl_seconds=heartbeat_ttl_seconds, time_provider=time_provider)
        self.protocol = protocol
        self._app_name = app_name
        self._aspect = aspect
        self._response_timeout_seconds = response_timeout_seconds
        self._announce_interval_seconds = announce_interval_seconds
        self._node_destinations: Dict[str, str] = {}
        self._node_identities: Dict[str, Any] = {}
        self._destination_to_node: Dict[str, str] = {}
        self._pending_responses: Dict[str, QueryResponse] = {}
        self._pending_events: Dict[str, threading.Event] = {}
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_no_interface_log = 0.0
        self._hash_mismatch_seen: Dict[str, str] = {}
        if rns_config_dir:
            RNS.Reticulum(rns_config_dir)
        else:
            RNS.Reticulum()
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        identity_dir = os.getenv("SWARM_IDENTITY_DIR") or os.path.join(base_dir, "storage", "identities")
        resolved_identity_path = identity_path or os.getenv("SWARM_IDENTITY_PATH")
        if not resolved_identity_path:
            resolved_identity_path = os.path.join(identity_dir, f"{node_id}.rns")
        try:
            os.makedirs(os.path.dirname(resolved_identity_path), exist_ok=True)
        except Exception:
            resolved_identity_path = None
        identity = None
        if resolved_identity_path and os.path.exists(resolved_identity_path):
            try:
                identity = RNS.Identity.from_file(resolved_identity_path)
            except Exception:
                identity = None
        if identity is None:
            identity = RNS.Identity()
            if resolved_identity_path:
                try:
                    identity.to_file(resolved_identity_path)
                except Exception:
                    pass
        self._identity = identity
        self._destination = RNS.Destination(
            self._identity,
            RNS.Destination.IN,
            RNS.Destination.SINGLE,
            self._app_name,
            self._aspect,
        )
        self._destination.set_packet_callback(self._on_packet)
        self._destination.set_link_established_callback(self._on_link_established)
        self._links: Dict[str, RNS.Link] = {}
        self._identity_hash_to_node_id: Dict[bytes, str] = {}
        
        self._announce_handler = _AnnounceHandler(self)
        RNS.Transport.register_announce_handler(self._announce_handler)
        self._destination_hash = self._destination.hash
        self._destination_hash_hex = self._destination_hash.hex()
        threading.Thread(target=self._announce_loop, daemon=True).start()

    def _has_outbound_interface(self) -> bool:
        for interface in list(RNS.Transport.interfaces):
            if getattr(interface, "OUT", False) and getattr(interface, "online", False):
                return True
        return False

    def is_node_reachable(self, node_id: str) -> bool:
        if node_id in self._workers:
            return True
        if not self._has_outbound_interface():
            return False
        if node_id not in self._node_identities:
            return False
        return self._ensure_path(node_id, wait_seconds=0)

    def filter_reachable_nodes(self, node_ids: List[str]) -> List[str]:
        return [node_id for node_id in node_ids if self.is_node_reachable(node_id)]

    def get_node_health(self, node_id: str) -> Dict[str, Any]:
        stored_hash = self._node_destinations.get(node_id)
        identity = self._node_identities.get(node_id)
        actual_hash = None
        hash_match = None
        if identity is not None:
            destination = RNS.Destination(
                identity,
                RNS.Destination.OUT,
                RNS.Destination.SINGLE,
                self._app_name,
                self._aspect,
            )
            actual_hash = destination.hash.hex()
            if stored_hash:
                hash_match = stored_hash == actual_hash
        has_path = False
        next_hop = None
        if stored_hash:
            destination_hash_bytes = bytes.fromhex(stored_hash)
            if RNS.Transport.has_path(destination_hash_bytes):
                next_hop = RNS.Transport.next_hop_interface(destination_hash_bytes)
                has_path = next_hop is not None and getattr(next_hop, "OUT", False)
        return {
            "node_id": node_id,
            "has_identity": identity is not None,
            "has_outbound_interface": self._has_outbound_interface(),
            "stored_hash": stored_hash,
            "actual_hash": actual_hash,
            "hash_match": hash_match,
            "has_path": has_path,
            "next_hop": getattr(next_hop, "name", None) if next_hop else None,
        }

    def announce(self, announcement: AnnounceCapabilities) -> None:
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
        if not self._has_outbound_interface():
            return
        heartbeat = Heartbeat(
            node_id=node_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._time_provider())),
            status="ok",
        )
        payload = self.protocol.encode_binary(heartbeat)
        for remote_node_id in list(self._node_identities.keys()):
            self._send_packet(remote_node_id, payload)

    def send_query(self, node_id: str, request: QueryRequest) -> QueryResponse:
        worker = self._workers.get(node_id)
        if worker is not None:
            return super().send_query(node_id, request)
        if node_id not in self._node_identities:
            raise ValueError(f"Unknown remote node: {node_id}")
        constraints = dict(request.constraints)
        constraints["target_node_id"] = node_id
        constraints["reply_to"] = self._destination_hash_hex
        constraints["reply_to_node_id"] = self.node_id
        network_request = QueryRequest(
            query_id=request.query_id,
            question=request.question,
            domain=request.domain,
            recursion_budget=request.recursion_budget,
            constraints=constraints,
        )
        payload = self.protocol.encode_binary(network_request)
        event = threading.Event()
        with self._pending_lock:
            self._pending_events[request.query_id] = event
        if not self._send_payload(node_id, payload):
            with self._pending_lock:
                self._pending_events.pop(request.query_id, None)
                self._pending_responses.pop(request.query_id, None)
            raise TimeoutError(f"No path to remote node {node_id}")
        event.wait()
        with self._pending_lock:
            response = self._pending_responses.pop(request.query_id, None)
            self._pending_events.pop(request.query_id, None)
        if response is None:
            raise TimeoutError(f"No response for query {request.query_id}")
        return response

    def send_task(self, node_id: str, assignment: TaskAssignment) -> TaskResult:
        worker = self._workers.get(node_id)
        if worker is not None:
            print(f"[NetworkTransport] Local delivery for Task {assignment.task.task_id} to {node_id}")
            return super().send_task(node_id, assignment)
        
        if node_id not in self._node_identities:
            raise ValueError(f"Unknown remote node: {node_id}")
        if not self._has_outbound_interface():
            raise ConnectionError("No outbound interfaces available for remote task delivery")
            
        print(f"[NetworkTransport] Remote delivery for Task {assignment.task.task_id} to {node_id} (via Reticulum)")
        
        # Update assignment with sender info if missing
        if not assignment.sender_node_id:
            assignment.sender_node_id = self.node_id
        if not assignment.sender_hash:
            assignment.sender_hash = self._destination_hash_hex
            
        payload = self.protocol.encode_binary(assignment)
        
        link = self._ensure_link_active(node_id, wait_seconds=2.0, attempts=3)
        if not link:
            raise ConnectionError(f"Link to {node_id} failed")

        event = threading.Event()
        with self._pending_lock:
            self._pending_events[assignment.assignment_id] = event
            
        print(f"[Transport] Sending Task {assignment.task.task_id} to {node_id} ({len(payload)} bytes)...")
        if len(payload) > link.MDU:
            print(f"[Transport] Using Resource for large payload ({len(payload)} bytes)")
            try:
                RNS.Resource(payload, link)
            except Exception as e:
                print(f"[Transport] Resource creation failed: {e}")
                with self._pending_lock:
                    self._pending_events.pop(assignment.assignment_id, None)
                raise
        else:
            packet = RNS.Packet(link, payload)
            try:
                packet.send()
            except Exception as e:
                print(f"[Transport] Packet send failed: {e}")
                with self._pending_lock:
                    self._pending_events.pop(assignment.assignment_id, None)
                raise

        print(f"[NetworkTransport] Waiting for response for Task {assignment.task.task_id}...")
        event.wait()
            
        with self._pending_lock:
            response = self._pending_responses.pop(assignment.assignment_id, None)
            self._pending_events.pop(assignment.assignment_id, None)
            
        if response is None:
             raise TimeoutError(f"No response for task {assignment.assignment_id}")
             
        if not isinstance(response, TaskResult):
            raise TypeError(f"Expected TaskResult, got {type(response)}")
            
        print(f"[NetworkTransport] Received valid response for Task {assignment.task.task_id} from {node_id}")
        return response

    def send_message(self, node_id: str, text: str, sender: Optional[str] = None) -> bool:
        if node_id in self._workers:
            return super().send_message(node_id, text, sender=sender)
        if node_id not in self._node_identities:
            return False
        message = TextMessage(
            message_id=str(uuid.uuid4()),
            sender=sender or self.node_id,
            recipient=node_id,
            text=text,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._time_provider())),
        )
        payload = self.protocol.encode_binary(message)
        if not self._send_payload(node_id, payload):
            return False
        self.emit_activity(
            "message_sent",
            node_id=node_id,
            payload={
                "message_id": message.message_id,
                "sender": message.sender,
                "recipient": message.recipient,
                "text": message.text,
            },
        )
        return True

    def send_context_update(self, node_id: str, content: str, context_id: Optional[str] = None) -> bool:
        if node_id in self._workers:
            return super().send_context_update(node_id, content, context_id=context_id)
        if node_id not in self._node_identities:
            return False
        context_id_value = context_id or str(uuid.uuid4())
        chunks = self._build_context_chunks(content, context_id_value)
        for chunk in chunks:
            update = ContextUpdate(
                context_id=context_id_value,
                content=chunk,
                source_node=self.node_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._time_provider())),
            )
            payload = self.protocol.encode_binary(update)
            if not self._send_payload(node_id, payload):
                return False
        return True

    def interface_status(self) -> List[Dict[str, Any]]:
        interfaces = []
        for interface in list(RNS.Transport.interfaces):
            name = getattr(interface, "name", None) or str(interface)
            interfaces.append(
                {
                    "name": name,
                    "type": interface.__class__.__name__,
                    "in": bool(getattr(interface, "IN", False)),
                    "out": bool(getattr(interface, "OUT", False)),
                    "online": bool(getattr(interface, "online", False)),
                    "bitrate": int(getattr(interface, "bitrate", 0) or 0),
                }
            )
        return interfaces

    def _announce_loop(self) -> None:
        while not self._stop_event.is_set():
            for announcement in list(self._announcements.values()):
                if announcement.node_id in self._workers:
                    self.announce(announcement)
            time.sleep(self._announce_interval_seconds)

    def _get_or_create_link(self, node_id: str) -> Optional[RNS.Link]:
        if node_id in self._links:
            link = self._links[node_id]
            if link.status == RNS.Link.ACTIVE:
                return link
            if link.status == RNS.Link.PENDING:
                return link
            if link.status == RNS.Link.CLOSED:
                self._links.pop(node_id, None)

        identity = self._node_identities.get(node_id)
        if not identity:
            print(f"[Transport] No identity for {node_id}, cannot create link.")
            return None
            
        destination = RNS.Destination(
            identity,
            RNS.Destination.OUT,
            RNS.Destination.SINGLE,
            self._app_name,
            self._aspect
        )
        
        print(f"[Transport] Establishing link to {node_id}...")
        link = RNS.Link(destination)
        link.set_link_closed_callback(self._on_link_closed)
        link.set_packet_callback(self._on_link_packet)
        link.set_resource_strategy(RNS.Link.ACCEPT_ALL)
        link.set_resource_concluded_callback(self._on_resource_concluded)
        
        self._links[node_id] = link
        return link

    def _ensure_link_active(self, node_id: str, wait_seconds: float = 2.0, attempts: int = 3) -> Optional[RNS.Link]:
        if not self._ensure_path(node_id, wait_seconds=wait_seconds):
            return None
        for attempt in range(max(1, attempts)):
            link = self._get_or_create_link(node_id)
            if not link:
                time.sleep(min(0.2, 0.05 * (attempt + 1)))
                continue
            deadline = time.monotonic() + wait_seconds
            while link.status == RNS.Link.PENDING and time.monotonic() < deadline:
                time.sleep(0.05)
            if link.status == RNS.Link.ACTIVE:
                return link
            if link.status == RNS.Link.CLOSED:
                self._links.pop(node_id, None)
            time.sleep(min(0.5, 0.1 * (attempt + 1)))
        return None

    def _on_link_established(self, link: RNS.Link) -> None:
        link.set_link_closed_callback(self._on_link_closed)
        link.set_packet_callback(self._on_link_packet)
        link.set_resource_strategy(RNS.Link.ACCEPT_ALL)
        link.set_resource_concluded_callback(self._on_resource_concluded)
        
        remote_identity = link.get_remote_identity()
        if remote_identity and remote_identity.hash in self._identity_hash_to_node_id:
             node_id = self._identity_hash_to_node_id[remote_identity.hash]
             self._links[node_id] = link
             print(f"[Transport] Link established with {node_id}")
        else:
             print(f"[Transport] Link established with unknown identity")

    def _on_link_closed(self, link: RNS.Link) -> None:
        for node_id, l in list(self._links.items()):
            if l == link:
                self._links.pop(node_id, None)
                print(f"[Transport] Link closed with {node_id}")
                break

    def _on_link_packet(self, data: bytes, packet: RNS.Packet) -> None:
        try:
            message = self.protocol.decode_binary(data)
        except Exception:
             try:
                 message = self.protocol.decode_compact(data)
             except Exception:
                 return
        self._handle_message(message)

    def _on_resource_concluded(self, resource: RNS.Resource) -> None:
        if resource.status == RNS.Resource.COMPLETE:
             try:
                 data = resource.data.read()
                 message = self.protocol.decode_binary(data)
                 self._handle_message(message)
             except Exception as e:
                 print(f"[Transport] Failed to decode resource data: {e}")

    def _on_packet(self, payload: bytes, packet: RNS.Packet) -> None:
        try:
            message = self.protocol.decode_binary(payload)
        except Exception:
            try:
                message = self.protocol.decode_compact(payload)
            except Exception:
                return
        
        # Handle Announce specially as it uses packet details
        if isinstance(message, AnnounceCapabilities):
            if message.destination_hash and message.node_id not in self._node_identities:
                self._node_destinations[message.node_id] = message.destination_hash
                self._destination_to_node[message.destination_hash] = message.node_id
                
                # Only request path if we don't have a stable link or path
                should_check_path = True
                if message.node_id in self._links:
                     link = self._links[message.node_id]
                     if link.status == RNS.Link.ACTIVE:
                         should_check_path = False
                
                if should_check_path:
                    self._ensure_path(message.node_id, wait_seconds=0)
            super().announce(message)
            return

        self._handle_message(message)

    def _handle_message(self, message: Any) -> None:
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
            reply_to_node_id = message.constraints.get("reply_to_node_id")
            if reply_to_node_id or reply_to:
                payload = self.protocol.encode_binary(response)
                if reply_to_node_id and reply_to_node_id in self._node_identities:
                    self._send_payload(reply_to_node_id, payload)
                elif reply_to and reply_to in self._destination_to_node:
                    self._send_payload(self._destination_to_node[reply_to], payload)
            return
        if isinstance(message, TaskAssignment):
            worker = None
            target = message.assigned_to_node
            if target in self._workers:
                worker = self._workers[target]
            elif self._workers:
                worker = next(iter(self._workers.values()))
            
            if worker:
                result = worker.handle_task(message)
                # Send result back
                payload = self.protocol.encode_binary(result)
                
                target_node_id = message.sender_node_id
                if target_node_id:
                    link = self._ensure_link_active(target_node_id, wait_seconds=1.5, attempts=2)
                    if link:
                        print(f"[Transport] Sending TaskResult to {target_node_id} via Link ({len(payload)} bytes)")
                        if len(payload) > link.MDU:
                            try:
                                RNS.Resource(payload, link)
                                return
                            except Exception as e:
                                print(f"[Transport] Failed to send result resource: {e}")
                        else:
                            try:
                                RNS.Packet(link, payload).send()
                                return
                            except Exception as e:
                                print(f"[Transport] Failed to send result packet: {e}")
                    if self._send_payload(target_node_id, payload):
                        return

                if message.sender_hash:
                    if message.sender_node_id and message.sender_node_id in self._node_identities:
                        self._send_payload(message.sender_node_id, payload)
                    elif message.sender_hash in self._destination_to_node:
                        node_id = self._destination_to_node[message.sender_hash]
                        self._send_payload(node_id, payload)
                    else:
                        pass
            return

        if isinstance(message, TaskResult):
            with self._pending_lock:
                if message.assignment_id in self._pending_events:
                    self._pending_responses[message.assignment_id] = message
                    self._pending_events[message.assignment_id].set()
            return

        if isinstance(message, QueryResponse):
            with self._pending_lock:
                event = self._pending_events.get(message.query_id)
                if event is None:
                    return
                self._pending_responses[message.query_id] = message
                event.set()
            return
        if isinstance(message, TextMessage):
            self.emit_activity(
                "message_received",
                node_id=message.sender,
                payload={
                    "message_id": message.message_id,
                    "sender": message.sender,
                    "recipient": message.recipient,
                    "text": message.text,
                },
            )
            return
        if isinstance(message, ContextUpdate):
            self._handle_context_update(message.content, message.source_node, message.context_id)
            return

    def _ensure_path(self, node_id: str, wait_seconds: float = 2.0) -> bool:
        destination_hash = self._node_destinations.get(node_id)
        if not destination_hash:
            return False
        destination_hash_bytes = bytes.fromhex(destination_hash)
        if RNS.Transport.has_path(destination_hash_bytes):
            next_hop_interface = RNS.Transport.next_hop_interface(destination_hash_bytes)
            if next_hop_interface is not None and getattr(next_hop_interface, "OUT", False):
                return True
        RNS.Transport.request_path(destination_hash_bytes)
        if wait_seconds <= 0:
            return False
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if RNS.Transport.has_path(destination_hash_bytes):
                next_hop_interface = RNS.Transport.next_hop_interface(destination_hash_bytes)
                if next_hop_interface is not None and getattr(next_hop_interface, "OUT", False):
                    return True
            time.sleep(0.1)
        return False

    def _send_packet(self, node_id: str, payload: bytes) -> bool:
        if not self._has_outbound_interface():
            now = time.monotonic()
            if now - self._last_no_interface_log > 5.0:
                print("[Transport] No outbound interfaces available")
                self._last_no_interface_log = now
            return False
        identity = self._node_identities.get(node_id)
        if identity is None:
            print(f"[Transport] Error: No identity for {node_id}")
            return False
        if not self._ensure_path(node_id):
            print(f"[Transport] Error: No path to {node_id}")
            return False
        destination = RNS.Destination(
            identity,
            RNS.Destination.OUT,
            RNS.Destination.SINGLE,
            self._app_name,
            self._aspect,
        )
        
        # Verify hash
        stored_hash = self._node_destinations.get(node_id)
        if stored_hash and destination.hash.hex() != stored_hash:
            mismatch_key = f"{destination.hash.hex()}!= {stored_hash}"
            if self._hash_mismatch_seen.get(node_id) != mismatch_key:
                self._hash_mismatch_seen[node_id] = mismatch_key
                print(f"[Transport] Warning: Destination hash mismatch for {node_id}! {destination.hash.hex()} != {stored_hash}")
            if stored_hash in self._destination_to_node and self._destination_to_node.get(stored_hash) == node_id:
                self._destination_to_node.pop(stored_hash, None)
            self._node_destinations[node_id] = destination.hash.hex()
            self._destination_to_node[destination.hash.hex()] = node_id
            
        print(f"[Transport] Sending packet to {node_id} ({len(payload)} bytes)...")
        if len(payload) > 465: # Approximate safe limit for RNS single packets
             print(f"[Transport] Warning: Payload size {len(payload)} exceeds typical MTU (465 bytes). Packet may fail.")

        packet = RNS.Packet(destination, payload)
        try:
            packet.send()
            if not packet.sent:
                print(f"[Transport] Packet not sent (check interfaces/path)")
                RNS.Transport.request_path(destination.hash)
                return False
            return True
        except Exception as e:
             print(f"[Transport] Packet send failed: {e}")
             return False

    def _send_payload(self, node_id: str, payload: bytes) -> bool:
        if not self._has_outbound_interface():
            now = time.monotonic()
            if now - self._last_no_interface_log > 5.0:
                print("[Transport] No outbound interfaces available")
                self._last_no_interface_log = now
            return False
        link = self._ensure_link_active(node_id, wait_seconds=1.5, attempts=2)
        if link:
            if len(payload) > link.MDU:
                try:
                    RNS.Resource(payload, link)
                    return True
                except Exception as e:
                    print(f"[Transport] Resource send failed: {e}")
                    return False
            try:
                packet = RNS.Packet(link, payload)
                packet.send()
                return packet.sent
            except Exception as e:
                print(f"[Transport] Packet send failed: {e}")
                return False
        if len(payload) > 465:
            print(f"[Transport] Payload too large without link ({len(payload)} bytes)")
            return False
        return self._send_packet(node_id, payload)


class _AnnounceHandler:
    def __init__(self, transport: NetworkTransport) -> None:
        self.transport = transport
        self.aspect_filter = f"{transport._app_name}.{transport._aspect}"

    def received_announce(self, destination_hash: bytes, announced_identity: Any, app_data: Optional[bytes]) -> None:
        if not app_data:
            return
        try:
            message = self.transport.protocol.decode_compact(app_data)
        except Exception:
            return
        if isinstance(message, AnnounceCapabilities):
            if announced_identity is not None:
                self.transport._node_identities[message.node_id] = announced_identity
                self.transport._identity_hash_to_node_id[announced_identity.hash] = message.node_id
                destination = RNS.Destination(
                    announced_identity,
                    RNS.Destination.OUT,
                    RNS.Destination.SINGLE,
                    self.transport._app_name,
                    self.transport._aspect,
                )
                expected_hash = destination.hash.hex()
                if message.destination_hash and message.destination_hash != expected_hash:
                    mismatch_key = f"{message.destination_hash}!={expected_hash}"
                    if self.transport._hash_mismatch_seen.get(message.node_id) != mismatch_key:
                        self.transport._hash_mismatch_seen[message.node_id] = mismatch_key
                        print(
                            f"[Transport] Warning: Announce hash mismatch for {message.node_id}! "
                            f"{message.destination_hash} != {expected_hash}"
                        )
                previous_hash = self.transport._node_destinations.get(message.node_id)
                if previous_hash and previous_hash in self.transport._destination_to_node:
                    if self.transport._destination_to_node.get(previous_hash) == message.node_id:
                        self.transport._destination_to_node.pop(previous_hash, None)
                message.destination_hash = expected_hash
                self.transport._node_destinations[message.node_id] = expected_hash
                self.transport._destination_to_node[expected_hash] = message.node_id
            super(NetworkTransport, self.transport).announce(message)
