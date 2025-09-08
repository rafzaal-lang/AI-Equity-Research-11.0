import json
import os
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

ART_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts")).resolve()
ART_DIR.mkdir(parents=True, exist_ok=True)

def _hash(obj: Any) -> str:
    """Generate a hash for reproducible artifact IDs."""
    raw = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]

def save_artifact(kind: str, payload: Dict[str, Any]) -> str:
    """Save an artifact with assumptions and return artifact ID."""
    timestamp = datetime.utcnow().isoformat()
    payload_with_meta = {
        **payload,
        "_metadata": {
            "kind": kind,
            "created_at": timestamp,
            "version": "1.0"
        }
    }
    
    aid = f"{kind}-{uuid.uuid4().hex[:8]}-{_hash(payload)}"
    path = ART_DIR / f"{aid}.json"
    
    path.write_text(json.dumps(payload_with_meta, indent=2), encoding="utf-8")
    return aid

def load_artifact(aid: str) -> Optional[Dict[str, Any]]:
    """Load an artifact by ID."""
    path = ART_DIR / f"{aid}.json"
    if not path.exists():
        return None
    
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return None

def list_artifacts(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all artifacts, optionally filtered by kind."""
    artifacts = []
    
    for path in ART_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("_metadata", {})
            
            if kind is None or metadata.get("kind") == kind:
                artifacts.append({
                    "id": path.stem,
                    "kind": metadata.get("kind"),
                    "created_at": metadata.get("created_at"),
                    "size": path.stat().st_size
                })
        except (json.JSONDecodeError, IOError):
            continue
    
    return sorted(artifacts, key=lambda x: x.get("created_at", ""), reverse=True)

def delete_artifact(aid: str) -> bool:
    """Delete an artifact by ID."""
    path = ART_DIR / f"{aid}.json"
    if path.exists():
        try:
            path.unlink()
            return True
        except OSError:
            return False
    return False

