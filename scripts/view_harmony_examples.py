import chromadb

client = chromadb.PersistentClient(path='db/rag')
col = client.get_collection('harmony_examples')

print(f'Total records: {col.count()}')
print()

results = col.get(include=['documents', 'metadatas'])

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
    print(f'=== Record {i+1} ===')
    print(f'Component: {meta.get("component_type", "Unknown")}')
    print(f'Time: {meta.get("created_at", "Unknown")}')
    print(f'Source: {meta.get("source", "Unknown")}')
    print()
    print(doc[:400] if len(doc) > 400 else doc)
    print()
    print('-' * 50)
    print()
