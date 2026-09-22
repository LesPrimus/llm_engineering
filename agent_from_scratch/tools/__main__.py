# Python imports the package before running this, and the package imports every
# tool module, so by here each tool has registered itself on the server.
# It is started from here rather than from a ``__main__`` block in server.py:
# ``-m`` on that module would run it a second time as ``__main__``, building a
# second server with none of the tools on it.
from .server import server

server.run()
