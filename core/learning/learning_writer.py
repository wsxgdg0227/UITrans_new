import re
import shortuuid
from datetime import datetime
from typing import Tuple

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
import torch


class LearningWriter:
    """极简增量学习写入器"""

    def __init__(self, persist_directory: str = "db/rag"):
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=False)
        )

        # 获取已存在的 collection 或创建新的（避免每次删除数据）
        try:
            self.collection = self.chroma_client.get_collection("harmony_examples")
        except:
            self.collection = self.chroma_client.create_collection(
                name="harmony_examples",
                metadata={
                    "description": "用户翻译示例库",
                    "hnsw:space": "cosine"  # 使用余弦相似度
                }
            )

        # 初始化 embedding function
        self._embedding_fn = HuggingFaceEmbeddings(
            model_name="models/embedding/bge-m3",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def write_if_needed(self, android_xml: str, arkui_code: str) -> Tuple[int, int]:
        """
        检查质量并写入

        Returns:
            (before_count, after_count)
        """
        before_count = self.collection.count()

        # 极简质量检查
        if len(arkui_code) < 30:
            return before_count, before_count
        if any(p in arkui_code for p in ["Error", "None", "null"]):
            return before_count, before_count

        # 生成唯一 ID
        doc_id = shortuuid.uuid()

        # 组合文档内容
        doc_content = f"""Android → HarmonyOS 翻译示例

Android 组件：
{android_xml}

ArkUI 组件：
{arkui_code}
"""
        # 手动计算 embedding
        embedding = self._embedding_fn.embed_documents([doc_content])[0]

        # 写入
        self.collection.add(
            ids=[doc_id],
            documents=[doc_content],
            embeddings=[embedding],
            metadatas=[{
                "source": "user-translation",
                "created_at": datetime.now().isoformat(),
                "component_type": self._extract_component_type(android_xml)
            }]
        )

        after_count = self.collection.count()
        return before_count, after_count

    def _extract_component_type(self, android_xml: str) -> str:
        match = re.search(r'<(\w+)', android_xml)
        return match.group(1) if match else "Unknown"


# 全局单例
_learning_writer = None


def get_learning_writer() -> LearningWriter:
    """获取全局 LearningWriter 实例"""
    global _learning_writer
    if _learning_writer is None:
        _learning_writer = LearningWriter()
    return _learning_writer
