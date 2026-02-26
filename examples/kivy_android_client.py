import os
import threading
import time
import uuid
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

from swarm import AnnounceCapabilities, NetworkTransport, Protocol, QueryResponse, VectorStore, Worker


class MobileWorker(Worker):
    def handle_query(self, request):
        return QueryResponse(
            query_id=request.query_id,
            claims=[f"node {self.transport.node_id} ok"],
            evidence=[],
            confidence=0.1,
            next_queries=[],
        )


class SwarmMobileApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.protocol = Protocol()
        self.node_id = os.getenv("SWARM_NODE_ID") or f"android-{uuid.uuid4().hex[:8]}"
        self.domain = os.getenv("SWARM_DOMAIN", "mobile")
        self.collections = [
            item.strip()
            for item in os.getenv("SWARM_COLLECTIONS", self.domain).split(",")
            if item.strip()
        ]
        rns_config_dir = os.getenv("RNS_CONFIG_DIR")
        self.transport = NetworkTransport(node_id=self.node_id, protocol=self.protocol, rns_config_dir=rns_config_dir)
        self.worker = MobileWorker(self.protocol, self.transport, VectorStore(collection_id=self.domain))
        self.transport.register_worker(self.node_id, self.worker)
        self.transport.announce(
            AnnounceCapabilities(
                node_id=self.node_id,
                domains=[self.domain],
                collections=self.collections,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                signature="kivy",
            )
        )
        self.stop_event = threading.Event()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def build(self):
        root = BoxLayout(orientation="vertical", spacing=8, padding=12)
        self.node_label = Label(text=f"Node: {self.node_id}  Domain: {self.domain}", size_hint_y=None, height=32)
        self.status_label = Label(text="ready", size_hint_y=None, height=28)
        self.nodes_label = Label(text="", size_hint_y=None, valign="top")
        self.nodes_label.bind(texture_size=self._resize_nodes_label)
        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(self.nodes_label)
        self.target_input = TextInput(hint_text="Target node id", size_hint_y=None, height=40, multiline=False)
        self.message_input = TextInput(hint_text="Message", size_hint_y=None, height=40, multiline=False)
        self.send_button = Button(text="Send", size_hint_y=None, height=40)
        self.refresh_button = Button(text="Refresh", size_hint_y=None, height=40)
        self.send_button.bind(on_press=self._send_message)
        self.refresh_button.bind(on_press=self._refresh_now)
        root.add_widget(self.node_label)
        root.add_widget(self.status_label)
        root.add_widget(scroll)
        root.add_widget(self.target_input)
        root.add_widget(self.message_input)
        root.add_widget(self.send_button)
        root.add_widget(self.refresh_button)
        Clock.schedule_interval(self._refresh_status, 1.0)
        return root

    def _resize_nodes_label(self, *_):
        self.nodes_label.text_size = (self.nodes_label.width, None)
        self.nodes_label.height = self.nodes_label.texture_size[1]

    def _format_nodes(self):
        nodes = self.transport.live_status()
        if not nodes:
            return "No nodes discovered"
        lines = ["Node | Domains | Collections | Last seen (s)"]
        for item in nodes:
            lines.append(
                f"{item['node_id']} | {','.join(item['domains'])} | {','.join(item['collections'])} | {item['last_seen_seconds']:.1f}"
            )
        return "\n".join(lines)

    def _refresh_status(self, *_):
        self.nodes_label.text = self._format_nodes()

    def _refresh_now(self, *_):
        self._refresh_status()

    def _send_message(self, *_):
        target = self.target_input.text.strip()
        text = self.message_input.text.strip()
        if not target or not text:
            self.status_label.text = "target and message required"
            return
        ok = self.transport.send_message(target, text, sender=self.node_id)
        self.status_label.text = "sent" if ok else "send failed"
        if ok:
            self.message_input.text = ""

    def _heartbeat_loop(self):
        while not self.stop_event.is_set():
            self.transport.heartbeat(self.node_id)
            time.sleep(5.0)

    def on_stop(self):
        self.stop_event.set()


if __name__ == "__main__":
    SwarmMobileApp().run()
