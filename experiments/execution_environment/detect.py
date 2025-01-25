import socket

from execution_environment.david import David
from execution_environment.environment import Environment
from execution_environment.metacentrum import Metacentrum
from execution_environment.pro import Pro


def detect_environment() -> Environment:
    match socket.gethostname():
        case 'Pro.local':
            return Pro()
        case name if name.startswith('david'):
            return David()
        case _:
            return Metacentrum()
