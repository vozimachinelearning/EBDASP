# Swarm Reticulum

## Descripción
Protocolo y base de código para un enjambre de agentes ultraligeros sobre Reticulum con RAG recursivo distribuido.

## Estructura
- `src/swarm`: núcleo del protocolo y orquestación
- `examples`: ejemplos de uso (incluye TUI)
- `start-reticulum.bat`: bootstrap de entorno, modelos y Reticulum
- `config`: configuración de Reticulum
- `models`: modelos descargados localmente

## Mensajes y serialización
- JSON compacto con `encode_compact` y `decode_compact` en `Protocol`
- Binario comprimido con `encode_binary` y `decode_binary`
- Routing con `AnnounceCapabilities`, `RouteRequest`, `RouteResponse`

## Iniciar Reticulum

```bat
c:\Users\LoOper\.reticulum\swarm\EBDASP\start-reticulum.bat
```

## Modelos
- Configura los repos con variables de entorno opcionales:
  - `SWARM_LLM_REPO` (por defecto `Qwen/Qwen2.5-0.5B-Instruct`)
  - `SWARM_EMBED_REPO` (por defecto `BAAI/bge-m3`)
- Rutas locales creadas por el .bat:
  - `SWARM_LLM_PATH` en `EBDASP\models\llm`
  - `SWARM_EMBED_PATH` en `EBDASP\models\embeddings`
- El demo detecta automáticamente `models\llm` y `models\embeddings` y expone:
  - `SWARM_MODEL_PATH`
  - `SWARM_EMBEDDING_PATH`

## Auto-conexión de dispositivos
- Usa AutoInterface en el config y el mismo `group_id` en todos los nodos
- Conecta los dispositivos al mismo WiFi
- Desactiva AP isolation en el router si está habilitado

## Ejemplo mínimo

```python
from swarm import AnnounceCapabilities, Coordinator, Orchestrator, Protocol, Transport, VectorStore, Worker

protocol = Protocol()
transport = Transport(node_id="coordinator")
coordinator = Coordinator(protocol, transport, VectorStore(collection_id="root"))
orchestrator = Orchestrator(coordinator, transport)

class WorkerDemo(Worker):
    def handle_query(self, request):
        return super().handle_query(request)

worker_a = WorkerDemo(protocol, transport, VectorStore(collection_id="historia"))
worker_b = WorkerDemo(protocol, transport, VectorStore(collection_id="ciencia"))

transport.register_worker("worker-a", worker_a)
transport.register_worker("worker-b", worker_b)

transport.announce(
    AnnounceCapabilities(
        node_id="worker-a",
        domains=["historia"],
        collections=["historia"],
        timestamp="2026-02-25T00:00:00Z",
        signature="firma-a",
    )
)
transport.announce(
    AnnounceCapabilities(
        node_id="worker-b",
        domains=["ciencia"],
        collections=["ciencia"],
        timestamp="2026-02-25T00:00:00Z",
        signature="firma-b",
    )
)

responses = orchestrator.distribute(
    question="Resumen de contexto",
    domain="historia",
    recursion_budget=1,
    max_workers=2,
)
```

## Ejecutar ejemplo

```bash
python c:\Users\LoOper\.reticulum\swarm\EBDASP\examples\minimal_demo.py
```

## Variables de entorno útiles
- `SWARM_NETWORK`: `1` para red Reticulum, `0` para modo local
- `SWARM_NODE_ID`: identificador del nodo
- `SWARM_DOMAIN`: dominio principal del nodo
- `SWARM_COLLECTIONS`: colecciones separadas por coma
- `RNS_CONFIG_DIR`: ruta del `config` de Reticulum
- `SWARM_MODEL_PATH` y `SWARM_EMBEDDING_PATH`: rutas locales de modelos

```bash
python c:\Users\LoOper\.reticulum\swarm\EBDASP\examples\minimal_demo.py
```
