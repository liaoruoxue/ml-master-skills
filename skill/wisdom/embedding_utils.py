#!/usr/bin/env python3
"""
ML-Master L3 Embedding Utilities
使用 sentence-transformers 实现语义检索

用法:
    python3 embedding_utils.py search "image classification plant disease"
    python3 embedding_utils.py add task_001 image_classification "Multi-label classification..." "wisdom/task_wisdom.md#task_001"
    python3 embedding_utils.py list
"""

import json
import sys
from pathlib import Path

# 配置
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = Path(__file__).parent / "embeddings.json"
DEFAULT_THRESHOLD = 0.4  # 降低阈值以提高召回率
DEFAULT_TOP_K = 3
KEYWORD_THRESHOLD_MULTIPLIER = 0.25  # 关键词匹配使用更低阈值 (0.4 * 0.25 = 0.1)

# 全局模型缓存
_model = None
_use_embeddings = None


def check_dependencies():
    """检查依赖是否可用"""
    global _use_embeddings
    if _use_embeddings is not None:
        return _use_embeddings

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _use_embeddings = True
    except ImportError:
        _use_embeddings = False
        print("[ML-Master] WARNING: sentence-transformers not installed.")
        print("[ML-Master] Install with: pip install sentence-transformers")
        print("[ML-Master] Falling back to keyword matching.")

    return _use_embeddings


def get_model():
    """延迟加载模型"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[ML-Master] Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("[ML-Master] Model loaded successfully.")
    return _model


def load_embeddings():
    """加载现有的嵌入索引"""
    if EMBEDDINGS_FILE.exists():
        try:
            return json.loads(EMBEDDINGS_FILE.read_text())
        except json.JSONDecodeError:
            print("[ML-Master] WARNING: embeddings.json corrupted, starting fresh")
    return {"model": MODEL_NAME, "entries": []}


def save_embeddings(data):
    """保存嵌入索引"""
    EMBEDDINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def add_wisdom(task_id: str, task_type: str, descriptor: str, wisdom_ref: str):
    """
    P2 Promotion: 添加新的任务智慧到向量索引

    Args:
        task_id: 唯一任务标识 (e.g., "plant-pathology-2021")
        task_type: 任务类型 (e.g., "image_classification", "tabular", "nlp")
        descriptor: 任务描述文本 (用于生成嵌入)
        wisdom_ref: 智慧引用路径 (e.g., "wisdom/task_wisdom.md#section")
    """
    data = load_embeddings()

    entry = {
        "id": task_id,
        "task_type": task_type,
        "descriptor": descriptor,
        "wisdom_ref": wisdom_ref,
        "keywords": extract_keywords(descriptor)  # 始终保存关键词作为后备
    }

    # 如果支持嵌入，生成向量
    if check_dependencies():
        model = get_model()
        embedding = model.encode(descriptor).tolist()
        entry["embedding"] = embedding

    # 更新或添加
    updated = False
    for i, existing in enumerate(data["entries"]):
        if existing["id"] == task_id:
            data["entries"][i] = entry
            updated = True
            break

    if not updated:
        data["entries"].append(entry)

    save_embeddings(data)
    print(f"[ML-Master] Added/updated wisdom for: {task_id}")
    return entry


def search_similar(query: str, top_k: int = DEFAULT_TOP_K, threshold: float = DEFAULT_THRESHOLD):
    """
    Context Prefetching: 搜索相似任务的智慧

    Args:
        query: 查询文本 (当前任务描述)
        top_k: 返回前 k 个结果
        threshold: 相似度阈值 (0-1)

    Returns:
        list: 相似任务的列表，包含 id, task_type, similarity, wisdom_ref
    """
    data = load_embeddings()

    if not data["entries"]:
        print("[ML-Master] No wisdom entries found in index.")
        return []

    results = []

    if check_dependencies():
        import numpy as np
        model = get_model()
        query_emb = model.encode(query)

        for entry in data["entries"]:
            if "embedding" not in entry:
                continue

            emb = np.array(entry["embedding"])
            # Cosine similarity
            similarity = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))

            if similarity >= threshold:
                results.append({
                    "id": entry["id"],
                    "task_type": entry["task_type"],
                    "similarity": round(similarity, 4),
                    "wisdom_ref": entry["wisdom_ref"],
                    "descriptor": entry.get("descriptor", "")[:100] + "..."
                })
    else:
        # 降级：关键词匹配
        query_keywords = set(extract_keywords(query))

        for entry in data["entries"]:
            entry_keywords = set(entry.get("keywords", extract_keywords(entry.get("descriptor", ""))))

            if not entry_keywords:
                continue

            # Jaccard similarity
            intersection = len(query_keywords & entry_keywords)
            union = len(query_keywords | entry_keywords)
            similarity = intersection / union if union > 0 else 0

            # 关键词匹配使用更低阈值 (DEFAULT_THRESHOLD * KEYWORD_THRESHOLD_MULTIPLIER)
            if similarity >= threshold * KEYWORD_THRESHOLD_MULTIPLIER:
                results.append({
                    "id": entry["id"],
                    "task_type": entry["task_type"],
                    "similarity": round(similarity, 4),
                    "wisdom_ref": entry["wisdom_ref"],
                    "descriptor": entry.get("descriptor", "")[:100] + "..."
                })

    # 按相似度排序
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def extract_keywords(text: str) -> list:
    """从文本提取关键词 (降级方案用)"""
    import re
    # 简单的关键词提取：小写、去标点、过滤停用词
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
                 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
                 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 'this', 'that', 'these', 'those', 'it', 'its'}

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in stopwords]


def list_entries():
    """列出所有已索引的智慧条目"""
    data = load_embeddings()

    if not data["entries"]:
        print("[ML-Master] No wisdom entries found.")
        return []

    print(f"[ML-Master] Found {len(data['entries'])} wisdom entries:")
    print("-" * 60)

    entries = []
    for entry in data["entries"]:
        has_embedding = "embedding" in entry
        print(f"  ID: {entry['id']}")
        print(f"  Type: {entry['task_type']}")
        print(f"  Ref: {entry['wisdom_ref']}")
        print(f"  Embedding: {'Yes' if has_embedding else 'No (keywords only)'}")
        print("-" * 60)
        entries.append(entry)

    return entries


def delete_entry(task_id: str):
    """删除指定的智慧条目"""
    data = load_embeddings()

    original_count = len(data["entries"])
    data["entries"] = [e for e in data["entries"] if e["id"] != task_id]

    if len(data["entries"]) < original_count:
        save_embeddings(data)
        print(f"[ML-Master] Deleted wisdom entry: {task_id}")
        return True
    else:
        print(f"[ML-Master] Entry not found: {task_id}")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands:")
        print("  search <query>           Search for similar wisdom")
        print("  add <id> <type> <desc> <ref>  Add wisdom to index")
        print("  list                     List all indexed entries")
        print("  delete <id>              Delete an entry")
        print("  check                    Check if dependencies are installed")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: embedding_utils.py search <query>")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        results = search_similar(query)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    elif cmd == "add":
        if len(sys.argv) < 6:
            print("Usage: embedding_utils.py add <task_id> <task_type> <descriptor> <wisdom_ref>")
            sys.exit(1)
        add_wisdom(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    elif cmd == "list":
        list_entries()

    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: embedding_utils.py delete <task_id>")
            sys.exit(1)
        delete_entry(sys.argv[2])

    elif cmd == "check":
        if check_dependencies():
            print("[ML-Master] All dependencies installed. Using semantic embeddings.")
            # 测试加载模型
            get_model()
        else:
            print("[ML-Master] Using fallback keyword matching.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
