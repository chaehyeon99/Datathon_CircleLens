from django.urls import path
from . import views

urlpatterns = [
    path('', views.index , name = "index"),
    path('survey', views.survey , name = 'survey'),
    path('search', views.search, name = 'search'),
    path('keyword', views.keyword, name= 'keyword'),
    path('reward', views.reward, name="reward"),
    path('login', views.login, name="login"),
    path('survey_result', views.survey_result, name="survey_result"),
    path('reward_result', views.reward_result, name="reward_result"),
    path('search_result', views.search_result, name="search_result"),
    path('keyword_result', views.keyword_result, name="keyword_result"),
]