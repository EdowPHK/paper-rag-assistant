import unittest
import sys
import types
from unittest.mock import patch


sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

qdrant_client = types.ModuleType("qdrant_client")
qdrant_models = types.ModuleType("qdrant_client.models")


class FakeDistance:
    DOT = "Dot"


class FakeVectorParams:
    def __init__(self, size: int, distance: str) -> None:
        self.size = size
        self.distance = distance


class FakePointStruct:
    def __init__(self, id: str, vector: list[float], payload: dict) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


qdrant_models.Distance = FakeDistance
qdrant_models.VectorParams = FakeVectorParams
qdrant_models.PointStruct = FakePointStruct
qdrant_client.QdrantClient = object
qdrant_client.models = qdrant_models
sys.modules.setdefault("qdrant_client", qdrant_client)
sys.modules.setdefault("qdrant_client.models", qdrant_models)

pymupdf = types.ModuleType("pymupdf")
sys.modules.setdefault("pymupdf", pymupdf)

import demo


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class FakeEncoder:
    tokenizer = FakeTokenizer()


class FakeRecord:
    def __init__(self, payload: dict) -> None:
        self.payload = payload


class FakeScrollClient:
    def __init__(self) -> None:
        self.calls = 0

    def scroll(
        self,
        collection_name: str,
        limit: int,
        offset: object,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[FakeRecord], object]:
        self.calls += 1
        return (
            [
                FakeRecord(
                    {
                        "source": "paper.pdf",
                        "chunk_id": "chunk-a",
                        "chunk_index": 0,
                        "page_start": 1,
                        "page_end": 1,
                        "heading": "Intro",
                        "text": "retrieval augmented generation",
                    }
                )
            ],
            None,
        )


class ChunkingTests(unittest.TestCase):
    def test_split_pdf_pages_into_chunks_uses_heading_and_token_window(self) -> None:
        pages = [
            {
                "source": "paper.pdf",
                "page_id": 1,
                "text": "1 Introduction\nalpha\nbeta\ngamma",
            },
            {
                "source": "paper.pdf",
                "page_id": 2,
                "text": "delta\nepsilon",
            },
        ]

        with patch.object(demo, "_get_encoder", return_value=FakeEncoder()):
            chunks = demo.split_pdf_pages_into_chunks(
                pages,
                target_tokens=12,
                overlap_tokens=2,
                model="fake-model",
            )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["source"] == "paper.pdf" for chunk in chunks))
        self.assertTrue(all(chunk["heading"] == "1 Introduction" for chunk in chunks))
        self.assertTrue(all(len(FakeTokenizer().encode(chunk["text"])) <= 12 for chunk in chunks))
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(isinstance(chunk["chunk_id"], str) for chunk in chunks))

    def test_split_pdf_pages_into_chunks_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            demo.split_pdf_pages_into_chunks([], target_tokens=10, overlap_tokens=10, model="fake-model")


class RetrievalTests(unittest.TestCase):
    def test_bm25_search_chunks_ranks_keyword_matches(self) -> None:
        chunks = [
            {
                "source": "paper.pdf",
                "chunk_id": "a",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "heading": "Introduction",
                "text": "retrieval augmented generation improves grounded answers",
            },
            {
                "source": "paper.pdf",
                "chunk_id": "b",
                "chunk_index": 1,
                "page_start": 2,
                "page_end": 2,
                "heading": "Experiments",
                "text": "optimizer learning rate batch size baseline",
            },
        ]

        results = demo.bm25_search_chunks("retrieval generation", chunks, top_k=2)

        self.assertEqual(results[0]["chunk_id"], "a")
        self.assertEqual(results[0]["retrieval_source"], "bm25")
        self.assertGreater(results[0]["score"], 0)

    def test_rrf_fuse_results_prefers_items_appearing_in_multiple_rankings(self) -> None:
        vector_results = [
            {"chunk_id": "a", "text": "vector only", "score": 0.90, "retrieval_source": "vector"},
            {"chunk_id": "b", "text": "shared", "score": 0.80, "retrieval_source": "vector"},
        ]
        bm25_results = [
            {"chunk_id": "b", "text": "shared", "score": 3.0, "retrieval_source": "bm25"},
            {"chunk_id": "c", "text": "keyword only", "score": 2.0, "retrieval_source": "bm25"},
        ]

        fused = demo.rrf_fuse_results([vector_results, bm25_results], top_k=3, k=10)

        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertEqual(fused[0]["retrieval_source"], "hybrid")
        self.assertIn("vector_score", fused[0])
        self.assertIn("bm25_score", fused[0])
        self.assertIn("rrf_score", fused[0])

    def test_list_indexed_pdf_chunks_reads_qdrant_payloads(self) -> None:
        fake_client = FakeScrollClient()

        with patch.object(demo, "_get_qdrant_client", return_value=fake_client):
            chunks = demo.list_indexed_pdf_chunks(collection_name="knowledge_base", limit=10)

        self.assertEqual(fake_client.calls, 1)
        self.assertEqual(chunks[0]["chunk_id"], "chunk-a")
        self.assertEqual(chunks[0]["text"], "retrieval augmented generation")
        self.assertEqual(chunks[0]["page_start"], 1)


if __name__ == "__main__":
    unittest.main()
