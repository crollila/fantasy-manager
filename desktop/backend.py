"""Frozen entry point. App data is provided by the desktop host, never bundled."""
import multiprocessing
import os
from pathlib import Path

if __name__ == '__main__':
    multiprocessing.freeze_support()
    os.environ.setdefault('FANTASY_DATA_DIR', str(Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Fantasy Manager' / 'data'))
    import copy
    import logging
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG
    # Timestamps in desktop.log, so a failure can be matched to what the user was doing.
    fmt = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(level=logging.WARNING, format=fmt)
    log_config = copy.deepcopy(LOGGING_CONFIG)
    for formatter in log_config['formatters'].values():
        formatter['fmt'] = fmt
        formatter.pop('use_colors', None)
    from app.api import app
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('FANTASY_PORT', '8000')), log_level='warning', log_config=log_config)
