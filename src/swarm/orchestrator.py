from typing import Any, Dict, List, Optional

from .coordinator import Coordinator
from .messages import QueryRequest, QueryResponse, RouteRequest
from .transport import Transport


class Orchestrator:
    def __init__(self, coordinator: Coordinator, transport: Transport) -> None:
        self.coordinator = coordinator
        self.transport = transport

    def distribute(
        self,
        question: str,
        domain: Optional[str] = None,
        recursion_budget: int = 2,
        constraints: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
    ) -> List[QueryResponse]:
        request = self.coordinator.build_query(
            question=question,
            domain=domain,
            recursion_budget=recursion_budget,
            constraints=constraints,
        )
        route_request = RouteRequest(query_id=request.query_id, domain=domain, limit=max_workers)
        route_response = self.transport.route(route_request)
        return [self.transport.send_query(node_id, request) for node_id in route_response.node_ids]

    def distribute_recursive(
        self,
        question: str,
        domain: Optional[str] = None,
        recursion_budget: int = 2,
        constraints: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
    ) -> List[QueryResponse]:
        root_request = self.coordinator.build_query(
            question=question,
            domain=domain,
            recursion_budget=recursion_budget,
            constraints=constraints,
        )
        pending: List[QueryRequest] = [root_request]
        responses: List[QueryResponse] = []
        while pending:
            request = pending.pop(0)
            if request.recursion_budget < 0:
                continue
            route_request = RouteRequest(query_id=request.query_id, domain=request.domain, limit=max_workers)
            route_response = self.transport.route(route_request)
            for node_id in route_response.node_ids:
                response = self.transport.send_query(node_id, request)
                responses.append(response)
                for next_request in response.next_queries:
                    if next_request.recursion_budget >= 0:
                        pending.append(next_request)
        return responses
