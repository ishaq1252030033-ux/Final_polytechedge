import sys
import os

project_home = '/home/polytechEDGE'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Ensure production env (optional)
os.environ.setdefault('FLASK_ENV', 'production')

# Import the Flask application object as 'application'
# NOTE: change 'edge' below if your package/module name differs
from edge import app as application
PY
