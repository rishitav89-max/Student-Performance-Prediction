import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.shortcuts import render
from django.db.models import Avg
from django.views.decorators.csrf import csrf_exempt

from .ml_service import predict_student_performance
from .models import Prediction

def validate_student_data(data):
    required_fields = [
        "student_id",
        "age",
        "gender",
        "attendance_percentage",
        "assignment_score",
        "internal_marks",
        "previous_percentage",
        "study_hours_per_day",
        "class_participation",
        "extracurricular_score"
    ]

    for field in required_fields:
        if field not in data:
            return f"Missing field: {field}"

    if not 14 <= data["age"] <= 18:
        return "Age must be between 14 and 18."

    numeric_fields = [
        "attendance_percentage",
        "assignment_score",
        "internal_marks",
        "previous_percentage",
        "study_hours_per_day",
        "extracurricular_score"
    ]

    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return f"{field} must be a number."

    if not 0 <= data["attendance_percentage"] <= 100:
        return "Attendance percentage must be between 0 and 100."

    if not 0 <= data["assignment_score"] <= 100:
        return "Assignment score must be between 0 and 100."

    if not 0 <= data["internal_marks"] <= 100:
        return "Internal marks must be between 0 and 100."

    if not 0 <= data["previous_percentage"] <= 100:
        return "Previous percentage must be between 0 and 100."

    if not 0.5 <= data["study_hours_per_day"] <= 8:
        return "Study hours must be between 0.5 and 8."

    if not 0 <= data["extracurricular_score"] <= 100:
        return "Extracurricular score must be between 0 and 100."

    if data["gender"] not in ["Male", "Female", "Other"]:
        return "Invalid gender."

    if data["class_participation"] not in ["Low", "Medium", "High"]:
        return "Invalid class participation."

    return None

@csrf_exempt
def predict_performance(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            # Validate data after receiving JSON data
            error = validate_student_data(data)
            if error:
                return JsonResponse({
                    "error": error
                }, status=400)

            student_data = {
                "age": data["age"],
                "gender": data["gender"],
                "attendance_percentage": data["attendance_percentage"],
                "assignment_score": data["assignment_score"],
                "internal_marks": data["internal_marks"],
                "previous_percentage": data["previous_percentage"],
                "study_hours_per_day": data["study_hours_per_day"],
                "class_participation": data["class_participation"],
                "extracurricular_score": data["extracurricular_score"]
            }

            result = predict_student_performance(student_data)

            Prediction.objects.create(
                student_id=data["student_id"],
                age=student_data["age"],
                gender=student_data["gender"],
                attendance_percentage=student_data["attendance_percentage"],
                assignment_score=student_data["assignment_score"],
                internal_marks=student_data["internal_marks"],
                previous_percentage=student_data["previous_percentage"],
                study_hours_per_day=student_data["study_hours_per_day"],
                class_participation=student_data["class_participation"],
                extracurricular_score=student_data["extracurricular_score"],
                predicted_percentage=result["predicted_percentage"],
                performance_category=result["performance_category"]
            )

            return JsonResponse(result)

        except KeyError as e:
            return JsonResponse({
                "error": f"Missing field: {e.args[0]}"
            }, status=400)

        except json.JSONDecodeError:
            return JsonResponse({
                "error": "Invalid JSON data."
            }, status=400)

    return JsonResponse({
        "error": "Only POST requests are allowed."
    }, status=405)


def get_predictions(request):
    
    if request.method == "GET":

        predictions = Prediction.objects.all().order_by("-created_at")

        data = []

        for prediction in predictions:
            data.append({
                "student_id": prediction.student_id,
                "predicted_percentage": prediction.predicted_percentage,
                "performance_category": prediction.performance_category,
                "created_at": prediction.created_at
            })

        return JsonResponse({
            "predictions": data
        })

    return JsonResponse({
        "error": "Only GET requests are allowed."
    }, status=405)


def prediction_history(request):
    
    student_id = request.GET.get("student_id", "").strip()

    predictions = Prediction.objects.all().order_by("-created_at")

    if student_id:
        predictions = predictions.filter(
            student_id__icontains=student_id
        )

    return render(
        request,
        "prediction/prediction_history.html",
        {
            "predictions": predictions,
            "student_id": student_id
        }
    )

def prediction_form(request):
    return render(
        request,
        "prediction/prediction_form.html"
    )


def model_performance(request):
    
    model_comparison = [
        {
            "model_name": "Linear Regression",
            "mae": 3.85,
            "rmse": 4.97,
            "r2_score": 0.882
        },
        {
            "model_name": "Decision Tree",
            "mae": 6.77,
            "rmse": 8.34,
            "r2_score": 0.667
        },
        {
            "model_name": "Random Forest",
            "mae": 4.60,
            "rmse": 5.68,
            "r2_score": 0.845
        }
    ]

    return render(
        request,
        "prediction/model_performance.html",
        {
            "model_comparison": model_comparison
        }
    )


def dashboard(request):
    
    total_predictions = Prediction.objects.count()

    average_predicted_percentage = Prediction.objects.aggregate(
        average=Avg("predicted_percentage")
    )["average"]

    if average_predicted_percentage is None:
        average_predicted_percentage = 0
    else:
        average_predicted_percentage = round(
        average_predicted_percentage,
        2
    )

    recent_predictions = Prediction.objects.all().order_by(
            "-created_at"
        )[:5]

    excellent_count = Prediction.objects.filter(
        performance_category="Excellent"
    ).count()

    good_count = Prediction.objects.filter(
        performance_category="Good"
    ).count()

    average_count = Prediction.objects.filter(
        performance_category="Average"
    ).count()

    at_risk_count = Prediction.objects.filter(
        performance_category="At Risk"
    ).count()


    # Calculate performance percentages

    if total_predictions > 0:

        excellent_percentage = round(
            (excellent_count / total_predictions) * 100,
            2
        )

        good_percentage = round(
            (good_count / total_predictions) * 100,
            2
        )

        average_percentage = round(
            (average_count / total_predictions) * 100,
            2
        )

        at_risk_percentage = round(
            (at_risk_count / total_predictions) * 100,
            2
        )

    else:

        excellent_percentage = 0
        good_percentage = 0
        average_percentage = 0
        at_risk_percentage = 0


    return render(
        request,
        "prediction/dashboard.html",
        {
            "total_predictions": total_predictions,
            "average_predicted_percentage": average_predicted_percentage,
            "excellent_count": excellent_count,
            "good_count": good_count,
            "average_count": average_count,
            "at_risk_count": at_risk_count,

            "excellent_percentage": excellent_percentage,
            "good_percentage": good_percentage,
            "average_percentage": average_percentage,
            "at_risk_percentage": at_risk_percentage,
            "recent_predictions": recent_predictions,
           
        }
    )


def prediction_detail(request, student_id):
    
    prediction = Prediction.objects.filter(
        student_id=student_id
    ).order_by("-created_at").first()

    if prediction is None:
        return render(
            request,
            "prediction/prediction_detail.html",
            {
                "prediction": None
            }
        )

    return render(
        request,
        "prediction/prediction_detail.html",
        {
            "prediction": prediction
        }
    )