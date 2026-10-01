import json
from pathlib import Path


class MemoryAgent:
	"""Persist analysis sessions and user preferences as local JSON."""

	def __init__(self, storage_path="data/memory.json"):
		self.storage_path = Path(storage_path)

	def _read(self):
		if not self.storage_path.exists():
			return {"sessions": {}, "preferences": {}}
		try:
			with self.storage_path.open("r", encoding="utf-8") as file:
				data = json.load(file)
		except (OSError, json.JSONDecodeError) as error:
			raise ValueError(f"Could not read memory file {self.storage_path}: {error}") from error
		data.setdefault("sessions", {})
		data.setdefault("preferences", {})
		return data

	def _write(self, data):
		self.storage_path.parent.mkdir(parents=True, exist_ok=True)
		with self.storage_path.open("w", encoding="utf-8") as file:
			json.dump(data, file, indent=2, default=str)

	def save_session(self, session_id, report):
		data = self._read()
		data["sessions"][session_id] = report
		self._write(data)
		return report

	def get_session(self, session_id):
		return self._read()["sessions"].get(session_id)

	def set_preference(self, key, value):
		data = self._read()
		data["preferences"][key] = value
		self._write(data)
		return value

	def get_preferences(self):
		return self._read()["preferences"]
