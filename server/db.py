# https://github.com/lars-tiede/aiorethink/blob/master/aiorethink/db.py
import threading
import rethinkdb as r
from errors import IllegalAccessError, AlreadyExistsError


class _OneConnPerThreadPool:
    """Keeps track of one RethinkDB connection per thread.

    Get (or create) the current thread's connection with get() or just
    __await__. close() closes and discards the the current thread's connection
    so that a subsequent __await__ or get opens a new connection.
    """

    def __init__(self):
        self._tl = threading.local()
        self._connect_kwargs = None

    def configure_db_connection(self, **connect_kwargs):
        if self._connect_kwargs != None:
            raise AlreadyExistsError("Can not re-configure DB connection(s)")
        self._connect_kwargs = connect_kwargs

    def __await__(self):
        return self.get().__await__()

    async def get(self):
        """Gets or opens the thread's DB connection.
        """
        if self._connect_kwargs == None:
            raise IllegalAccessError("DB connection parameters not set yet")

        if not hasattr(self._tl, "conn") or not self._tl.conn.is_open():
            self._tl.conn = await r.connect(**self._connect_kwargs)

        return self._tl.conn

    async def close(self, noreply_wait=True):
        """Closes the thread's DB connection.
        """
        if hasattr(self._tl, "conn"):
            if self._tl.conn.is_open():
                await self._tl.conn.close(noreply_wait)
            del self._tl.conn


async def init_pool(app):
    print('init pool')
    r.set_loop_type("asyncio")

    conf = app['config']['rethinkDb']

    pool = _OneConnPerThreadPool()
    pool.configure_db_connection(**conf)
    app['db'] = pool


async def close_pool(app):
    print('close pool')
    pool = app['db']
    await pool.close()
