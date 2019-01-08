from aiohttp import web
from routes import setup_routes
from settings import config
from db import init_pool, close_pool

app = web.Application()
setup_routes(app)
app['config'] = config
app.on_startup.append(init_pool)
app.on_cleanup.append(close_pool)
web.run_app(app, port=5000)