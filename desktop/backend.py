"""Frozen entry point. App data is provided by the desktop host, never bundled."""
import multiprocessing
import os
from pathlib import Path

if __name__ == '__main__':
    multiprocessing.freeze_support()
    os.environ.setdefault('FANTASY_DATA_DIR', str(Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Fantasy Manager' / 'data'))
    import uvicorn
    from app.api import app
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('FANTASY_PORT', '8000')), log_level='warning')
