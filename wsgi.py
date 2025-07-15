# wsgi.py
import gevent.monkey

from app import create_app

gevent.monkey.patch_all()

application = create_app()
