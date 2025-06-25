from django.urls import path
from .views import views, api_views

urlpatterns = [
    path('iso_calc/', api_views.CalcISOAPIView.as_view(), name='iso_calc'),
]
