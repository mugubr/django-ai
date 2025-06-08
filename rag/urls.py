from django.urls import path
from . import views

urlpatterns = [
    path('dar_contexto_ia/', views.dar_contexto_ia, name='dar_contexto_ia'),
    path('chat/', views.chat, name='chat'),
    path('stream_response/', views.stream_response, name='stream_response'),
    path('ver_fontes/<int:id>', views.ver_fontes, name='ver_fontes'),
]

