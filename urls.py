from django.urls import path
from .views import (
    predict_performance,
    get_predictions,
    prediction_form,
    prediction_history,
    dashboard,
    prediction_detail,
    model_performance
)

urlpatterns = [
    path("", prediction_form, name="prediction_form"),
    path("predict/", predict_performance, name="predict_performance"),
    path("predictions/", get_predictions, name="get_predictions"),
    path("history/", prediction_history, name="prediction_history"),
    path("dashboard/", dashboard, name="dashboard"),
    path(
    "prediction/<str:student_id>/",
    prediction_detail,
    name="prediction_detail"
),
    path("model-performance/", model_performance, name="model_performance"),
]   