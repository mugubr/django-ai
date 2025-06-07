from django.shortcuts import render, redirect

from rolepermissions.checkers import has_permission
from django.http import Http404
from .models import Treinamentos
from django_q.models import Task

def treinar_ia(request):
    if not has_permission(request.user, 'treinar_ia'):
        raise Http404()
    if request.method == 'GET':
        tasks = Task.objects.all()
        return render(request, 'treinar_ia.html', {'tasks': tasks})
    elif request.method == 'POST':
        site = request.POST.get('site')
        conteudo = request.POST.get('conteudo')
        documento = request.FILES.get('documento')

        treinamento = Treinamentos(
            site=site,
            conteudo=conteudo,
            documento=documento
        )

        treinamento.save()

        return redirect('treinar_ia')