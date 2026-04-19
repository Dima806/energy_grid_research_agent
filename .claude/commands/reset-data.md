Wipe corpus and vector store, regenerate, and re-ingest. Use after schema changes to GridFinding/ComplianceArtefact or prompt changes that require fresh embeddings.

```bash
make clean && make generate && make ingest
```
