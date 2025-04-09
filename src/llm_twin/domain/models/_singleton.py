import abc
import threading
import typing


class SingletonMeta(abc.ABCMeta):
    """
    Thread-safe Singleton implementation.
    """

    _lock: typing.ClassVar[threading.Lock] = threading.Lock()
    _instances: typing.ClassVar[dict[type, object]] = {}

    def __call__(cls, *args: object, **kwargs: object):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]
