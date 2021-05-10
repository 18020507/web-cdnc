from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from .views import (
    index,
    predictImage,
    editOutput,
)

urlpatterns = [
    url('^$', index, name='homepage'),
    url(r'^predictImage', predictImage, name='predict'),
    # url(r'^editOutput/(?P<filePath>\w+)', editOutput.as_view(), name='editOutput')
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)