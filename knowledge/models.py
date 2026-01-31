from django.db import models

class KnowledgeNode(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    tags = models.JSONField(default=list)  # ["ml", "optimization"]

    def __str__(self):
        return self.title

class KnowledgeEdge(models.Model):
    source = models.ForeignKey(KnowledgeNode, on_delete=models.CASCADE, related_name="out_edges")
    target = models.ForeignKey(KnowledgeNode, on_delete=models.CASCADE, related_name="in_edges")
    relation_type = models.CharField(max_length=50)  # prerequisite, related
    confidence = models.FloatField(default=1.0)

    def __str__(self):
        return f"{self.source} -> {self.target} ({self.relation_type})"
