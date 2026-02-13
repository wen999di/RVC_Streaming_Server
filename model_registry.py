import json
import os
from dataclasses import dataclass
from pathlib import Path
import uuid


def _safe_basename(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("invalid filename")
    name = os.path.basename(name.strip())
    if not name or name in (".", ".."):
        raise ValueError("invalid filename")
    if any(sep in name for sep in ("/", "\\", "\x00")):
        raise ValueError("invalid filename")
    return name


@dataclass(frozen=True)
class ModelSlot:
    slot: str
    title: str
    allowed_ext: tuple[str, ...]


SLOTS: tuple[ModelSlot, ...] = (
    ModelSlot(slot="hubert_base", title="Hubert 基模", allowed_ext=(".pt",)),
    ModelSlot(slot="rmvpe", title="RMVPE 声高模型", allowed_ext=(".pt",)),
    ModelSlot(slot="uvr5_weight", title="UVR5 模型", allowed_ext=(".pth", ".onnx")),
)


class ModelRegistry:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.path = self.base_dir / "model_registry.json"
        self._slots: dict[str, dict] = {}
        self._voice: dict = {"active_id": "", "models": []}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._slots = {}
            self._voice = {"active_id": "", "models": []}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                self._slots = {}
                self._voice = {"active_id": "", "models": []}
                return

            if "slots" in d or "voice_models" in d:
                slots = d.get("slots") or {}
                voice = d.get("voice_models") or {}
                if not isinstance(slots, dict):
                    slots = {}
                if not isinstance(voice, dict):
                    voice = {}
                self._slots = {str(k): v for k, v in slots.items() if isinstance(v, dict)}
                models = voice.get("models") if isinstance(voice.get("models"), list) else []
                self._voice = {
                    "active_id": str(voice.get("active_id") or ""),
                    "models": [
                        m
                        for m in models
                        if isinstance(m, dict)
                        and isinstance(m.get("id"), str)
                        and isinstance(m.get("name"), str)
                        and isinstance(m.get("pth"), str)
                    ],
                }
                return

            if all(isinstance(v, str) for v in d.values()):
                self._slots = {
                    str(k): {"files": [str(v)], "active": str(v)}
                    for k, v in d.items()
                    if isinstance(k, str) and isinstance(v, str) and v
                }
                self._voice = {"active_id": "", "models": []}
                self._save()
                return

            self._slots = {}
            self._voice = {"active_id": "", "models": []}
        except Exception:
            self._slots = {}
            self._voice = {"active_id": "", "models": []}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"slots": self._slots, "voice_models": self._voice},
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.path)

    def list_slots(self) -> dict:
        out: dict[str, dict] = {}
        for s in SLOTS:
            state = self._slots.get(s.slot) if isinstance(self._slots.get(s.slot), dict) else {}
            files = state.get("files") if isinstance(state.get("files"), list) else []
            files = [str(x) for x in files if isinstance(x, str) and x]
            active = str(state.get("active") or "")
            if active and active not in files:
                active = files[0] if files else ""
            out[s.slot] = {
                "title": s.title,
                "allowed_ext": list(s.allowed_ext),
                "files": files,
                "active": active,
            }
        return out

    def add_to_slot(self, *, slot: str, filename: str, files_dir: Path) -> dict:
        slot = str(slot).strip()
        slot_def = next((s for s in SLOTS if s.slot == slot), None)
        if not slot_def:
            raise ValueError("unknown_slot")
        filename = _safe_basename(filename)
        if slot_def.allowed_ext and not filename.lower().endswith(
            tuple(ext.lower() for ext in slot_def.allowed_ext)
        ):
            raise ValueError("invalid_extension")
        target = files_dir / filename
        if not target.exists() or not target.is_file():
            raise FileNotFoundError("file_not_found")

        state = self._slots.get(slot) if isinstance(self._slots.get(slot), dict) else {}
        files = state.get("files") if isinstance(state.get("files"), list) else []
        files = [str(x) for x in files if isinstance(x, str) and x]
        if filename not in files:
            files.append(filename)
        self._slots[slot] = {"files": files, "active": filename}
        self._save()
        return self.list_slots()[slot]

    def activate_in_slot(self, *, slot: str, filename: str) -> dict:
        slot = str(slot).strip()
        filename = _safe_basename(filename)
        slot_state = self.list_slots().get(slot)
        if not slot_state:
            raise ValueError("unknown_slot")
        if filename not in slot_state.get("files", []):
            raise ValueError("not_bound")
        self._slots[slot] = {"files": slot_state["files"], "active": filename}
        self._save()
        return self.list_slots()[slot]

    def remove_from_slot(self, *, slot: str, filename: str) -> dict:
        slot = str(slot).strip()
        filename = _safe_basename(filename)
        slot_state = self.list_slots().get(slot)
        if not slot_state:
            raise ValueError("unknown_slot")
        files = [x for x in slot_state.get("files", []) if x != filename]
        active = slot_state.get("active", "")
        if active == filename:
            active = files[0] if files else ""
        self._slots[slot] = {"files": files, "active": active}
        self._save()
        return self.list_slots()[slot]

    def set_slot(self, *, slot: str, filename: str, files_dir: Path) -> dict:
        return self.add_to_slot(slot=slot, filename=filename, files_dir=files_dir)

    def list_voice_models(self) -> dict:
        active_id = str(self._voice.get("active_id") or "")
        models = self._voice.get("models") if isinstance(self._voice.get("models"), list) else []
        out_models = []
        for m in models:
            if not isinstance(m, dict):
                continue
            model_id = str(m.get("id") or "")
            name = str(m.get("name") or "")
            pth = str(m.get("pth") or "")
            index = str(m.get("index") or "")
            if not model_id or not name or not pth:
                continue
            out_models.append(
                {"id": model_id, "name": name, "pth": pth, "index": index, "active": model_id == active_id}
            )
        return {"active_id": active_id, "models": out_models}

    def add_voice_model(self, *, name: str, pth: str, index: str, files_dir: Path) -> dict:
        name = str(name or "").strip()
        if not name:
            raise ValueError("invalid_name")
        pth = _safe_basename(pth)
        index = _safe_basename(index) if index else ""
        if not pth.lower().endswith(".pth"):
            raise ValueError("invalid_extension_pth")
        if index and not index.lower().endswith(".index"):
            raise ValueError("invalid_extension_index")
        if not (files_dir / pth).exists():
            raise FileNotFoundError("file_not_found_pth")
        if index and not (files_dir / index).exists():
            raise FileNotFoundError("file_not_found_index")

        model_id = str(uuid.uuid4())
        model = {"id": model_id, "name": name, "pth": pth, "index": index}
        models = self._voice.get("models") if isinstance(self._voice.get("models"), list) else []
        models = [m for m in models if isinstance(m, dict)]
        models.append(model)
        self._voice["models"] = models
        self._voice["active_id"] = model_id
        self._save()
        return self.list_voice_models()

    def activate_voice_model(self, *, model_id: str) -> dict:
        model_id = str(model_id or "").strip()
        models = self._voice.get("models") if isinstance(self._voice.get("models"), list) else []
        if not any(isinstance(m, dict) and str(m.get("id") or "") == model_id for m in models):
            raise ValueError("unknown_voice_model")
        self._voice["active_id"] = model_id
        self._save()
        return self.list_voice_models()

    def remove_voice_model(self, *, model_id: str) -> dict:
        model_id = str(model_id or "").strip()
        models = self._voice.get("models") if isinstance(self._voice.get("models"), list) else []
        models = [m for m in models if isinstance(m, dict) and str(m.get("id") or "") != model_id]
        self._voice["models"] = models
        if str(self._voice.get("active_id") or "") == model_id:
            self._voice["active_id"] = str(models[0].get("id")) if models else ""
        self._save()
        return self.list_voice_models()

    def rename_file_references(self, *, old_name: str, new_name: str) -> dict:
        old_name = _safe_basename(old_name)
        new_name = _safe_basename(new_name)
        if old_name.lower() == new_name.lower():
            return {"changed": False}

        changed = False

        for slot, state in list(self._slots.items()):
            if not isinstance(state, dict):
                continue
            files = state.get("files") if isinstance(state.get("files"), list) else []
            files2 = []
            for f in files:
                if not isinstance(f, str) or not f:
                    continue
                files2.append(new_name if f.lower() == old_name.lower() else f)
            active = str(state.get("active") or "")
            if active.lower() == old_name.lower():
                active = new_name
            if files2 != files or active != state.get("active"):
                self._slots[slot] = {"files": files2, "active": active}
                changed = True

        models = self._voice.get("models") if isinstance(self._voice.get("models"), list) else []
        for m in models:
            if not isinstance(m, dict):
                continue
            pth = str(m.get("pth") or "")
            index = str(m.get("index") or "")
            if pth.lower() == old_name.lower():
                m["pth"] = new_name
                changed = True
            if index and index.lower() == old_name.lower():
                m["index"] = new_name
                changed = True

        if changed:
            self._save()
        return {"changed": changed}
