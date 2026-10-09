from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):

    list_display = (
        "student_id",
        "predicted_percentage",
        "performance_category",
        "created_at",
    )

    list_filter = (
        "performance_category",
        "gender",
        "class_participation",
    )

    search_fields = (
        "student_id",
    )