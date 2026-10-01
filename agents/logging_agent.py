from datetime import datetime, timezone
import json
import uuid


class LoggingAgent:
	"""Collect structured events under a trace ID for each analysis run."""

	def __init__(self):
		self.events = []
		self.active_traces = {}

	def start_trace(self, name):
		trace_id = str(uuid.uuid4())
		self.active_traces[trace_id] = {
			"trace_id": trace_id,
			"name": name,
			"started_at": datetime.now(timezone.utc).isoformat(),
		}
		return trace_id

	def log_event(self, trace_id, event, details=None):
		if trace_id not in self.active_traces:
			raise ValueError(f"Unknown trace ID: {trace_id}")
		record = {
			"trace_id": trace_id,
			"event": event,
			"timestamp": datetime.now(timezone.utc).isoformat(),
		}
		if details is not None:
			record["details"] = details
		self.events.append(record)
		return record

	def end_trace(self, trace_id):
		if trace_id not in self.active_traces:
			raise ValueError(f"Unknown trace ID: {trace_id}")
		trace = self.active_traces.pop(trace_id)
		trace["ended_at"] = datetime.now(timezone.utc).isoformat()
		trace["events"] = [event for event in self.events if event["trace_id"] == trace_id]
		return trace

	def export(self):
		return json.dumps(self.events, indent=2, default=str)
