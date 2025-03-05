#!/bin/env python3
import sys
from typing import Literal
from logging import Logger, Handler, Formatter, StreamHandler, FileHandler


class My_logger(Logger):
    def __init__(self, name: str = "",
                 level: int | str = 0,
                 propagate=False,
                 handler: Handler | Literal['std', 'file', None] = 'std',
                 file=None, mode='a', _format=None
                 ) -> None:
        super().__init__(name, level)
        self.propagate = propagate
        self.remove_handlers()
        self.setLevel(level)
        if handler is not None:
            if isinstance(handler, Handler):
                self.addHandler(handler)
            else:
                self.add_handler(handler, level=level, file=file,
                                 mode=mode, _format=_format)

    def add_handler(self, _type: Literal["std", "file"] = 'std',
                    level=0, file=None, mode='a', _format=None) -> None:
        if _type == "std":
            handler = StreamHandler(sys.stdout)
        elif _type == "file":
            if file is None:
                file = 'log.txt'
            handler = FileHandler(str(file), mode=mode)
        else:
            raise ValueError('_type is error')
        handler.setLevel(level)
        if _format is None:
            _format = Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        elif isinstance(_format, str):
            _format = Formatter(_format)
        elif isinstance(_format, Formatter):
            pass
        else:
            raise TypeError('_format type is error')
        handler.setFormatter(_format)
        self.addHandler(handler)

    def add_handlers(self, handlers: list) -> None:
        for v in handlers:
            if isinstance(v, Handler):
                self.addHandler(v)
            else:
                raise ValueError(str(v) + "is not a logging.Handler!")

    def remove_handlers(self) -> None:
        for v in self.handlers:
            self.removeHandler(v)
