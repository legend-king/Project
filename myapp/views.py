from django.shortcuts import render,HttpResponse
from django.http import QueryDict

# Create your views here.
def index(request):
    if request.method=="POST":
        d = dict()
        d['sfh'] = request.POST.get('sfh')
        d['web_traffic'] = request.POST.get('web_traffic')
        d['popupwindow'] = request.POST.get('popupwindow')
        d['url_length'] = request.POST.get('url_length')
        d['ssl_finalstate'] = request.POST.get('ssl_finalstate')
        d['ageofdomain'] = request.POST.get('ageofdomain')
        d['request_url']=request.POST.get('request_url')
        d['having_ip_address']=request.POST.get('having_ip_address')
        d['url_of_anchor']=request.POST.get('url_of_anchor')
        print(d)
    return render(request, 'index.html')