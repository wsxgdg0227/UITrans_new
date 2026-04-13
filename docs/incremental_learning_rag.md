# UITrans 增量学习 RAG 功能优化报告

## 1. 背景与目标

### 1.1 问题描述

在演示场景中，需要向导师展示"数据库能自动学习"的功能。当用户完成一次成功的 Android→HarmonyOS 组件翻译后，系统应自动将这次翻译结果存入知识库，以便后续翻译时能够检索参考。

### 1.2 优化目标

1. **写入层**：翻译成功后自动将 Android XML + ArkUI 代码对存入独立的 ChromaDB collection
2. **检索层**：翻译时同时检索官方组件文档库和用户翻译示例库
3. **可视化**：输出翻译结果时显示知识库增长情况

### 1.3 设计约束

- **不修改 ui.py 的 UI 结构**：避免引入新的 Tab 死锁问题
- **最小改动原则**：只在必要位置添加逻辑
- **独立 collection**：用户翻译示例与官方文档分离存储

---

## 2. 架构设计

### 2.1 数据流架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户输入 Android XML                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     convert_android_component_to_harmony()            │
│                     (ui.py - 组件翻译函数)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CodeMonkeyAgent.translate_component_v1()          │
│                     (code_monkey.py - 翻译主流程)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                      ▼
        ┌────────────────────────────────┐  ┌────────────────────────────────┐
        │   _query_component_document()   │  │   _query_user_examples()       │
        │   (查询官方组件文档库)           │  │   (查询用户翻译示例库)          │
        │   harmony-component-doc         │  │   harmony_examples             │
        │   3658 条记录                   │  │   N 条记录                     │
        └────────────────────────────────┘  └────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │              组合最终组件文档（含用户示例）                    │
        │   "# HarmonyOS组件\n..."                                   │
        │   "\n\n# 用户翻译示例 (来自知识库)\n..."                        │
        └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │              LLM 生成 ArkUI 代码                           │
        └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │              LearningWriter.write_if_needed()               │
        │              (写入用户翻译示例到 harmony_examples)           │
        └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │              返回翻译结果 + "知识库由 X 变成 X+1"            │
        └─────────────────────────────────────────────────────────────┘
```

### 2.2 数据库结构

```
ChromaDB (db/rag/chroma.sqlite3)
│
├── harmony-component-doc (collection)
│   ├── 用途：HarmonyOS 官方组件文档库
│   ├── 记录数：3658 条
│   ├── Embedding 模型：bge-m3 (1024 维)
│   ├── 距离度量：cosine
│   └── 示例文档：
│       "组件名：TextInput
│        组件描述：TextInput 是ArkUI中的文本输入组件..."
│
└── harmony_examples (collection)
    ├── 用途：用户翻译示例库（增量学习）
    ├── 记录数：0 → N（随使用增长）
    ├── Embedding 模型：bge-m3 (1024 维)
    ├── 距离度量：cosine
    └── 示例文档：
        "Android → HarmonyOS 翻译示例

         Android 组件：
         <EditText android:hint="请输入" ... />

         ArkUI 组件：
         TextInput({ placeholder: '请输入' })..."
```

---

## 3. 核心代码逻辑

### 3.1 LearningWriter - 用户翻译示例写入器

**文件位置**：`core/learning/learning_writer.py`

```python
class LearningWriter:
    """极简增量学习写入器"""

    def __init__(self, persist_directory: str = "db/rag"):
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=False)
        )
        # 创建独立的 collection
        self.collection = self.chroma_client.create_collection(
            name="harmony_examples",
            metadata={
                "description": "用户翻译示例库",
                "hnsw:space": "cosine"  # 使用余弦相似度
            }
        )
        # 初始化 embedding 函数
        self._embedding_fn = HuggingFaceEmbeddings(
            model_name="models/embedding/bge-m3",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def write_if_needed(self, android_xml: str, arkui_code: str) -> Tuple[int, int]:
        """
        检查质量并写入

        Returns:
            (before_count, after_count) - 写入前后的记录数
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

        # 手动计算 embedding（必须与查询时使用的模型一致）
        embedding = self._embedding_fn.embed_documents([doc_content])[0]

        # 写入 ChromaDB
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
```

**关键设计点**：

1. **独立 Collection**：`harmony_examples` 与官方 `harmony-component-doc` 分离
2. **手动 Embedding**：由于 ChromaDB 0.5.x 的 embedding function API 限制，采用手动计算后传入
3. **质量检查**：跳过长度过短或包含错误标记的翻译
4. **元数据**：记录来源、时间戳、组件类型

### 3.2 CodeMonkeyAgent - 扩展检索能力

**文件位置**：`core/pilot/code_monkey.py`

#### 3.2.1 初始化用户示例库客户端

```python
def __init__(self, llm_client: LLMClient, ...):
    # ... 原有代码 ...
    self.component_db_client = Chroma(...)  # harmony-component-doc

    # 新增：用户翻译示例库客户端
    self.examples_db_client = Chroma(
        collection_name="harmony_examples",
        persist_directory=ConfigLoader.get_config().rag_config.persist_directory,
        embedding_function=HuggingFaceEmbeddings(...),  # 与 component_db_client 相同
    )
```

#### 3.2.2 查询用户翻译示例

```python
def _query_user_examples(self, query_text: str, k: int = 3, score_threshold: float = 0.5) -> List[tuple]:
    """查询用户翻译示例库 harmony_examples

    Args:
        query_text: 查询文本（安卓组件描述）
        k: 返回数量
        score_threshold: 相似度阈值

    Returns:
        List of (document, metadata, score) tuples
    """
    try:
        # 检查 collection 是否有数据
        if self.examples_db_client._collection.count() == 0:
            return []

        # 使用 component_db_client 的 embedding function（保证维度一致）
        query_embedding = self.component_db_client._embedding_function.embed_documents([query_text])

        # 执行查询
        query_results = self.examples_db_client._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        # 解析结果
        results = []
        if query_results["documents"] and query_results["documents"][0]:
            for doc, meta, distance in zip(...):
                score = 1 - distance  # 转换为相似度
                if score >= score_threshold:
                    results.append((doc, meta, score))

        return results
    except Exception as e:
        logger.warning(f"查询用户示例库失败: {e}")
        return []
```

#### 3.2.3 集成到组件文档生成流程

在 `_query_component_document()` 方法中（约第 395-420 行）：

```python
# 3.6 添加用户翻译示例到文档
user_examples_document = ""
if user_examples_results:
    user_examples_document = "\n\n# 用户翻译示例 (来自知识库)\n"
    for doc, meta, score in user_examples_results:
        component_type = meta.get("component_type", "Unknown")
        created_at = meta.get("created_at", "Unknown")
        user_examples_document += f"\n## 翻译示例 (相似度: {score:.2f}, 类型: {component_type})\n"
        user_examples_document += doc + "\n"

return {
    "document": final_component_document + user_examples_document,
    "component_document": get_harmony_component(component_query.components),
    "queries": total_query_results,
    "components": component_query.components,
    "idea": component_query.idea,
    "user_examples": user_examples_results  # 供后续使用
}
```

### 3.3 UI 集成

**文件位置**：`ui.py` - `convert_android_component_to_harmony()` 函数

```python
def convert_android_component_to_harmony(android_xml):
    # ... 现有翻译逻辑（不变） ...

    translations, agent_state = code_monkey_agent.translate_component_v1(breakdown_android_layout)
    harmonyos_code = translations[0].target_component_code

    # ===== 增量学习写入 =====
    writer = get_learning_writer()
    before_count, after_count = writer.write_if_needed(android_xml, harmonyos_code)

    if after_count > before_count:
        status_line = f"\n\n知识库由 {before_count} 变成 {after_count}"
        harmonyos_code = harmonyos_code + status_line
    # ==============================

    return harmonyos_code
```

---

## 4. 执行流程

### 4.1 单次组件翻译完整流程

```
1. 用户输入
   └─ Android XML: <EditText android:hint="请输入文字" ... />

2. 翻译初始化
   └─ CodeMonkeyAgent 实例化
      ├─ component_db_client (harmony-component-doc)
      └─ examples_db_client (harmony_examples)

3. 组件文档查询 (_query_component_document)
   ├─ 3.1 查询示例文档 (harmony-component-doc)
   │  └─ similarity_search_with_relevance_scores()
   ├─ 3.2 查询组件文档 (harmony-component-doc)
   │  └─ _collection.query()
   └─ 3.3 新增：查询用户翻译示例 (_query_user_examples)
      └─ examples_db_client._collection.query()
         ├─ 嵌入查询文本
         ├─ 检索 harmony_examples
         └─ 返回 top-k 相似结果

4. LLM 生成
   └─ translate_android_component.prompt
      ├─ 官方组件文档
      ├─ 用户翻译示例 ← 新增
      └─ Android XML → ArkUI 代码

5. 增量学习写入
   └─ LearningWriter.write_if_needed()
      ├─ 质量检查（长度、错误标记）
      ├─ 生成文档内容
      ├─ 计算 embedding
      └─ 写入 harmony_examples

6. 返回结果
   └─ ArkUI 代码 + "知识库由 0 变成 1"
```

### 4.2 用户示例检索时序

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  CodeMonkey  │         │ examples_db_client│         │    ChromaDB       │
│    Agent     │         │  (Chroma Client)  │         │   (SQLite)        │
└──────────────┘         └──────────────────┘         └──────────────────┘
        │                        │                        │
        │ 1. _query_user_examples │                        │
        │───────────────────────>│                        │
        │                        │                        │
        │ 2. embed_documents([query])                     │
        │───────────────────────>│                        │
        │                        │                        │
        │ 3. query_embeddings=[embedding]                 │
        │                        │──────────────────────>│
        │                        │                        │
        │ 4. query_results                              │
        │                        │<───────────────────────│
        │                        │                        │
        │ 5. 解析结果 (1-distance=score)                │
        │<───────────────────────│                        │
        │                        │                        │
        │ 6. 返回 [(doc, meta, score), ...]             │
        │                        │                        │
```

---

## 5. 数据库查看方法

### 5.1 Python 脚本查看

```python
import chromadb

client = chromadb.PersistentClient(path="db/rag")

# 列出所有 collection
for col in client.list_collections():
    print(f"{col.name}: {client.get_collection(col.name).count()} 条")

# 查看 harmony_examples 内容
col = client.get_collection("harmony_examples")
print(f"\n总记录数: {col.count()}")

# 获取最近 3 条
results = col.get(limit=3, include=["documents", "metadatas"])
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"\n组件类型: {meta['component_type']}")
    print(f"时间: {meta['created_at']}")
    print(f"内容:\n{doc[:200]}...")
```

### 5.2 SQLite 直接查看

```bash
cd UITrans
sqlite3 db/rag/chroma.sqlite3

# 查看所有表
sqlite> .tables

# 查看 harmony_examples 记录数
sqlite> SELECT COUNT(*) FROM langchain_chromadb_collection_content
   ...> WHERE uuid IN (SELECT uuid FROM chroma_collection_name
   ...> WHERE name='harmony_examples');

# 查看最近添加的记录
sqlite> SELECT * FROM langchain_chromadb_collection_content
   ...> ORDER BY rowid DESC LIMIT 5;
```

---

## 6. 关键配置

### 6.1 ChromaDB 配置

```yaml
# config.yaml
rag:
  persist_directory: db/rag
  embedding:
    text:
      model: models/embedding/bge-m3  # 必须使用相同模型
```

### 6.2 检索参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `k` | 3 | 返回最多 3 条用户示例 |
| `score_threshold` | 0.5 | 相似度阈值（实际阈值 0.3-0.5 可调） |

---

## 7. 演示流程

### 7.1 演示步骤

1. **启动前**：确认 `harmony_examples` 记录数
   ```bash
   pdm run python -c "
   import chromadb
   client = chromadb.PersistentClient(path='db/rag')
   col = client.get_collection('harmony_examples')
   print(f'翻译前: {col.count()} 条')
   "
   ```

2. **启动 UI**
   ```bash
   cd UITrans
   pdm run python ui.py
   ```

3. **执行翻译**：在 "组件转译" Tab 中输入或选择示例组件

4. **观察输出**：翻译成功后，输出末尾显示
   ```
   知识库由 0 变成 1
   ```

5. **验证数据库**
   ```bash
   # 查询记录数变化
   pdm run python -c "
   import chromadb
   client = chromadb.PersistentClient(path='db/rag')
   col = client.get_collection('harmony_examples')
   print(f'翻译后: {col.count()} 条')
   "
   ```

---

## 8. 技术要点总结

### 8.1 ChromaDB Embedding 兼容性问题

**问题**：ChromaDB 0.5.x 要求 collection 的 embedding 维度必须一致。

**解决方案**：
1. 手动计算 embedding 并传入：`embeddings=[embedding]`
2. 查询时使用相同模型生成 query embedding
3. Collection 创建时指定 `hnsw:space: "cosine"` 保证距离度量一致

### 8.2 ChromaDB Client 复用

**问题**：`examples_db_client` 的 embedding function 与 `component_db_client` 必须完全一致。

**解决方案**：查询时直接使用 `component_db_client._embedding_function` 生成 query embedding。

### 8.3 质量过滤

**策略**：
- 长度检查：`len(arkui_code) < 30` → 跳过
- 错误检查：包含 `Error`, `None`, `null` → 跳过
- 后续可扩展：基于置信度分数、用户反馈等

---

## 9. 文件清单

### 9.1 新增文件

| 文件 | 说明 |
|------|------|
| `core/learning/__init__.py` | 模块初始化 |
| `core/learning/learning_writer.py` | 用户翻译示例写入器 |

### 9.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `core/pilot/code_monkey.py` | 添加 `examples_db_client` 和 `_query_user_examples()` |
| `ui.py` | 添加增量学习写入调用 |

### 9.3 新增文档

| 文件 | 说明 |
|------|------|
| `docs/incremental_learning_rag.md` | 本文档 |

---

## 10. 后续优化建议

1. **相似去重**：避免重复写入相似的翻译示例
2. **置信度过滤**：基于 LLM 输出的置信度决定是否写入
3. **用户反馈**：允许用户标记翻译质量
4. **Rerank 优化**：使用 bge-reranker-v2-m3 对检索结果重排
5. **增量更新**：对已有示例进行更新而非重复写入
