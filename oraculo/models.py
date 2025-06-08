from django.db import models

class Treinamentos(models.Model):
    site = models.URLField()
    conteudo = models.TextField()
    documento = models.FileField(upload_to='documentos')

    def __str__(self):
        return self.site

class DataTreinamentos(models.Model):
    metadata = models.JSONField (null=True, blank=True)
    textos = models.TextField()

class Pergunta(models.Model):
    data_treinamento = models.ManyToManyField(DataTreinamentos)
    pergunta = models.TextField()

    def __str__(self):
        return self.pergunta
