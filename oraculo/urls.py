from django.urls import path
from . import views

urlpatterns = [
    path('treinar_ia/', views.treinar_ia, name='treinar_ia'),

]