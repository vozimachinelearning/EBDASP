from __future__ import annotations

import threading
import uuid
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import RNS

from .messages import (
    AnnounceCapabilities,
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
        self._lock = threading.RLock()

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

    def _prune_stale(self) -> None:
        now = self._time_provider()
        expired: List[str] = []
        with self._lock:
            for node_id, last_seen in self._last_seen.items():
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
        self._node_identities: Dict[str, Any] = {}
        self._destination_to_node: Dict[str, str] = {}
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
        self._destination.set_link_established_callback(self._on_link_established)
        self._links: Dict[str, RNS.Link] = {}
        self._identity_hash_to_node_id: Dict[bytes, str] = {}
        
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
        if not self._send_packet(node_id, payload):
            with self._pending_lock:
                self._pending_events.pop(request.query_id, None)
                self._pending_responses.pop(request.query_id, None)
            raise TimeoutError(f"No path to remote node {node_id}")
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

    def send_task(self, node_id: str, assignment: TaskAssignment) -> TaskResult:
        worker = self._workers.get(node_id)
        if worker is not None:
            print(f"[NetworkTransport] Local delivery for Task {assignment.task.task_id} to {node_id}")
            return super().send_task(node_id, assignment)
        
        if node_id not in self._node_identities:
            raise ValueError(f"Unknown remote node: {node_id}")
            
        print(f"[NetworkTransport] Remote delivery for Task {assignment.task.task_id} to {node_id} (via Reticulum)")
        
        # Update assignment with sender info if missing
        if not assignment.sender_node_id:
            assignment.sender_node_id = self.node_id
        if not assignment.sender_hash:
            assignment.sender_hash = self._destination_hash_hex
            
        payload = self.protocol.encode_binary(assignment)
        
        link = self._get_or_create_link(node_id)
        if not link:
             raise ValueError(f"Cannot establish link to {node_id}")
        
        # Wait for link
        deadline = time.time() + 15
        while link.status != RNS.Link.ACTIVE:
             if link.status == RNS.Link.CLOSED:
                  raise ConnectionError(f"Link to {node_id} failed")
             time.sleep(0.1)
             if time.time() > deadline:
                  raise TimeoutError(f"Timeout connecting to {node_id}")

        event = threading.Event()
        with self._pending_lock:
            self._pending_events[assignment.assignment_id] = event
            
        print(f"[Transport] Sending Task {assignment.task.task_id} to {node_id} ({len(payload)} bytes)...")
        if len(payload) > link.MDU:
             print(f"[Transport] Using Resource for large payload ({len(payload)} bytes)")
             try:
                 resource = RNS.Resource(payload, link)
                 # Wait for resource to start transferring? 
                 # Resource is background. We just wait for response.
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
        if not event.wait(self._response_timeout_seconds * 2): # Double timeout for large transfers
            print(f"[NetworkTransport] Timeout waiting for Task {assignment.task.task_id} response")
            with self._pending_lock:
                self._pending_events.pop(assignment.assignment_id, None)
                self._pending_responses.pop(assignment.assignment_id, None)
            raise TimeoutError(f"No response for task {assignment.assignment_id}")
            
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
        if not self._send_packet(node_id, payload):
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
        link.start()
        return link

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
            if not message.destination_hash:
                message.destination_hash = packet.destination_hash.hex()
            if message.destination_hash:
                self._node_destinations[message.node_id] = message.destination_hash
                self._destination_to_node[message.destination_hash] = message.node_id
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
                    self._send_packet(reply_to_node_id, payload)
                elif reply_to and reply_to in self._destination_to_node:
                    self._send_packet(self._destination_to_node[reply_to], payload)
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
                
                # Try to send via Link first
                target_node_id = message.sender_node_id
                if target_node_id:
                     link = self._get_or_create_link(target_node_id)
                     if link:
                         print(f"[Transport] Sending TaskResult to {target_node_id} via Link ({len(payload)} bytes)")
                         if len(payload) > link.MDU:
                             try:
                                 RNS.Resource(payload, link)
                             except Exception as e:
                                 print(f"[Transport] Failed to send result resource: {e}")
                         else:
                             try:
                                 RNS.Packet(link, payload).send()
                             except Exception as e:
                                 print(f"[Transport] Failed to send result packet: {e}")
                         return

                # Fallback to connectionless
                # Determine destination
                if message.sender_hash:
                    # We can send to hash if we have a mapping or use raw RNS if possible.
                    # self._send_packet uses node_id.
                    # If we have sender_node_id and it's in identities, great.
                    if message.sender_node_id and message.sender_node_id in self._node_identities:
                        self._send_packet(message.sender_node_id, payload)
                    elif message.sender_hash in self._destination_to_node:
                        node_id = self._destination_to_node[message.sender_hash]
                        self._send_packet(node_id, payload)
                    else:
                        # We have the hash but no node_id mapping.
                        # _send_packet requires node_id to lookup identity.
                        # This is a limitation of current implementation.
                        # We'll log error or try best effort.
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
            print(f"[Transport] Warning: Destination hash mismatch! {destination.hash.hex()} != {stored_hash}")
            
        print(f"[Transport] Sending packet to {node_id} ({len(payload)} bytes)...")
        if len(payload) > 465: # Approximate safe limit for RNS single packets
             print(f"[Transport] Warning: Payload size {len(payload)} exceeds typical MTU (465 bytes). Packet may fail.")

        packet = RNS.Packet(destination, payload)
        try:
            packet.send()
            return True
        except Exception as e:
             print(f"[Transport] Packet send failed: {e}")
             return False


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
            if not message.destination_hash:
                message.destination_hash = destination_hash.hex()
            if message.destination_hash:
                self.transport._node_destinations[message.node_id] = message.destination_hash
                self.transport._destination_to_node[message.destination_hash] = message.node_id
            if announced_identity is not None:
                self.transport._node_identities[message.node_id] = announced_identity
                self.transport._identity_hash_to_node_id[announced_identity.hash] = message.node_id
            super(NetworkTransport, self.transport).announce(message)
