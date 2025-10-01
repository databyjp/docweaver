from src.docweaver.models import Task
with open("tasks/temp/weaviate-production-readiness-training/module_9_backup.md", "r") as f:
    src_material = f.read()

task = Task(
    objective="Document any missing information from this material",
    context=src_material,
    focus="Ensure the key material from this training module is covered completely, correctly and accurately",
)
