
from django.shortcuts import render
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# DB import removed — pages work without database


# -------------------- ADMIN LOGIN --------------------
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')

        print("User ID is =", usrid)

        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')

    return render(request, 'AdminLogin.html')


# -------------------- VIEW USERS --------------------
def ViewRegisteredUsers(request):
    # No DB query — return empty list so page loads without any table
    data = []
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


# -------------------- ACTIVATE USER --------------------
def AdminActivaUsers(request):
    # No DB query — activation skipped, page loads without any table
    data = []
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


# -------------------- ADMIN HOME --------------------
def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


# -------------------- FILE UPLOAD --------------------
@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')

        if file:
            file_path = 'media/' + file.name

            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            return JsonResponse({
                "message": "File uploaded successfully",
                "filename": file.name
            })

        return JsonResponse({"error": "No file found"})

    return JsonResponse({"error": "Invalid request method"})

