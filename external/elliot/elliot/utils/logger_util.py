import atexit
import logging
from logging.config import ConvertingList, ConvertingDict, valid_ident
from logging.handlers import QueueHandler, QueueListener
from queue import Queue


def _resolve_handlers(l):
    if not isinstance(l, ConvertingList):
        return l

    # Indexing the list performs the evaluation.
    return [l[i] for i in range(len(l))]


def _resolve_queue(q):
    if not isinstance(q, ConvertingDict):
        return q
    if '__resolved_value__' in q:
        return q['__resolved_value__']

    cname = q.pop('class')
    klass = q.configurator.resolve(cname)
    props = q.pop('.', None)
    kwargs = {k: q[k] for k in q if valid_ident(k)}
    result = klass(**kwargs)
    if props:
        for name, value in props.items():
            setattr(result, name, value)

    q['__resolved_value__'] = result
    return result


class QueueListenerHandler(logging.Handler):
    """Handler that feeds a QueueListener (forwarding to `handlers`) through a queue.

    Wraps a QueueHandler instead of subclassing it: since Python 3.12,
    logging.config.dictConfig special-cases every QueueHandler subclass and
    rejects this config ("Unable to configure handler 'queue_listener'").
    Behaviour is unchanged on older Pythons.
    """

    def __init__(self, handlers, respect_handler_level=False, auto_run=True, queue=None):
        super().__init__()
        queue = _resolve_queue(queue) if queue is not None else Queue(-1)
        self.queue = queue
        self._queue_handler = QueueHandler(queue)
        handlers = _resolve_handlers(handlers)
        self._listener = QueueListener(
            queue,
            *handlers,
            respect_handler_level=respect_handler_level)
        self._running = False
        if auto_run:
            self.start()
            atexit.register(self.stop)

    def start(self):
        if not self._running:
            self._listener.start()
            self._running = True

    def stop(self):
        if self._running:
            self._listener.stop()
            self._running = False

    def emit(self, record):
        self._queue_handler.emit(record)