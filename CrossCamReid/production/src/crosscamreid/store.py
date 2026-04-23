from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class SIDStore:
    def __init__(
        self,
        db_path: str,
        collection: str,
        dim: int,
        fresh: bool,
        max_embeddings_per_sid: int,
    ):
        self.collection = collection
        self.max_embeddings_per_sid = max_embeddings_per_sid
        path = Path(db_path)

        if fresh and path.exists():
            print(f"[Qdrant] Wiping existing DB: {db_path}")
            shutil.rmtree(path)

        print(f"[Qdrant] Opening local DB: {db_path} (collection={collection})")
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=db_path)

        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            print(f"[Qdrant] Created collection with dim={dim}")

        self._next_sid = self._compute_next_sid()
        self._counts = self._compute_counts()
        print(f"[Qdrant] Next SID: {self._next_sid} (existing SIDs: {len(self._counts)})")

    def _compute_next_sid(self) -> int:
        max_sid = 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                sid = int(point.payload.get("sid", 0))
                if sid > max_sid:
                    max_sid = sid
            if offset is None:
                break
        return max_sid + 1

    def _compute_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                sid = int(point.payload.get("sid", 0))
                counts[sid] = counts.get(sid, 0) + 1
            if offset is None:
                break
        return counts

    def search_top1(self, embedding: np.ndarray) -> tuple[int | None, float]:
        if self._next_sid <= 1:
            return None, 0.0

        response = self.client.query_points(
            collection_name=self.collection,
            query=embedding.tolist(),
            limit=1,
            with_payload=True,
        )
        hits = response.points
        if not hits:
            return None, 0.0
        top = hits[0]
        return int(top.payload["sid"]), float(top.score)

    def append(self, sid: int, embedding: np.ndarray) -> None:
        if self._counts.get(sid, 0) >= self.max_embeddings_per_sid:
            return

        self.client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"sid": sid},
                )
            ],
        )
        self._counts[sid] = self._counts.get(sid, 0) + 1

    def new_sid(self, embedding: np.ndarray) -> int:
        sid = self._next_sid
        self._next_sid += 1
        self.append(sid, embedding)
        return sid

    def total_sids(self) -> int:
        return self._next_sid - 1

