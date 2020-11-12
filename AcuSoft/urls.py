"""AcuSoft URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from AcuSoft.views import home, RT60, RT60_resp, Paneles, Paneles_resp, Modos, Modos_resp
  

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('RT60/', RT60),
    path('RT60_resp/',RT60_resp),
    path('Paneles/', Paneles),
    path('Paneles_resp/', Paneles_resp),
    path('Modos/', Modos),
    path('Modos_resp/', Modos_resp)
]
