from django.db import models

class Prediction(models.Model):

    student_id = models.CharField(max_length=20)

    age = models.IntegerField()

    gender = models.CharField(max_length=10)

    attendance_percentage = models.FloatField()

    assignment_score = models.FloatField()

    internal_marks = models.FloatField()

    previous_percentage = models.FloatField()

    study_hours_per_day = models.FloatField()

    class_participation = models.CharField(max_length=10)

    extracurricular_score = models.FloatField()

    predicted_percentage = models.FloatField()

    performance_category = models.CharField(max_length=20)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.student_id