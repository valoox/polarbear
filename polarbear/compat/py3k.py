from abc import ABCMeta
import six


def abstract(*args):
    """Builds an abstract base class with the provided base"""
    class Abstract(six.with_metaclass(ABCMeta, *args)):
        """Abstract base class"""
        pass
    return Abstract
