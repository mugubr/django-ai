from django.conf import settings
from django.shortcuts import render, redirect

from rolepermissions.checkers import has_permission
from django.http import Http404
from .models import DataTreinamentos, Pergunta, Treinamentos
from django_q.models import Task
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from pathlib import Path
from django.http import StreamingHttpResponse

def dar_contexto_ia(request):
    if not has_permission(request.user, 'dar_contexto_ia'):
        raise Http404()
    if request.method == 'GET':
        tasks = Task.objects.all()
        return render(request, 'dar_contexto_ia.html', {'tasks': tasks})
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

        return redirect('dar_contexto_ia')
    
@csrf_exempt
def chat(request):
    if request.method == 'GET':
        return render(request, 'chat.html')
    elif request.method == 'POST':
        pergunta_user = request.POST.get('pergunta')

        pergunta = Pergunta(
            pergunta=pergunta_user
        )
        pergunta.save()

        return JsonResponse({'id': pergunta.id})

@csrf_exempt
def stream_response(request):
    id_pergunta = request.POST.get('id_pergunta')
    pergunta = Pergunta.objects.get(id=id_pergunta)
    def stream_generator():
        embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        vectordb = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

        docs = vectordb.similarity_search(pergunta.pergunta, k=5)
        for doc in docs:
            dt = DataTreinamentos.objects.create(
                metadata=doc.metadata,
                textos=doc.page_content
            )
            dt.save()
            pergunta.data_treinamento.add(dt)

        contexto = "\n\n".join([
            f"Material: {doc.page_content}"
            for doc in docs
        ])

        messages = [
            {"role": "system", "content": f"Você é um assistente virtual e deve responder com precissão as perguntas sobre uma empresa.\n\n{contexto}"},
            {"role": "user", "content": pergunta.pergunta}
        ]

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            streaming=True,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

        for chunk in llm.stream(messages):
            token = chunk.content
            if token:
                yield token

    return StreamingHttpResponse(stream_generator(), content_type='text/plain; charset=utf-8')

def ver_fontes(request, id):
    pergunta = Pergunta.objects.get(id=id)
    return render(request, 'ver_fontes.html', {'pergunta': pergunta})