# The server is started from here rather than from a ``__main__`` block in
# server.py: ``-m`` on that module would import it once through the package
# and then run it a second time as ``__main__``, building a second server.
from .server import server

server.run()
