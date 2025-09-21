import random

from ..operators import MUT_GAUSS


class Allele:
    """
    An allele.
    :var _min: The minimum value of the allel.
    :var _max: The maximum value of the allel.
    """

    # values to adjust in the subclass
    _min = 0  # the minimum value of the allel
    _max = 100  # the maximum value of the allel

    __value = None  # the value of the allel

    def __init__(self, value=None, normalized=False):
        """
        Creates an allele with the given value.
        :param value: The value of the allel.
        :param normalized: Flag whether the passed value is normalized.
        :type normalized: bool
        """
        if value is None:
            self.set(random.random(), normalized=True)  # Set a random value if no value is given
        else:
            self.set(value, normalized=normalized)

    def set(self, value, normalized=False):
        """
        Sets the value of the allel.
        :param value: The value of the allel.
        :param normalized: Flag whether the passed value is normalized.
        :type normalized: bool
        """
        # denormalize the value if necessary
        if normalized:
            value = value * (self.__class__.get_max() - self.__class__.get_min()) + self.__class__.get_min()

        # check if the value is in the range
        if not self.__class__.get_min() <= value <= self.__class__.get_max():
            value = min(max(value, self.__class__.get_min()), self.__class__.get_max())

        self.__value = value

    @classmethod
    def get_min(cls):
        """
        :return: The minimum value that the allele can have.
        :rtype: float
        """
        return cls._min

    @classmethod
    def get_max(cls):
        """
        :return: The maximum value that the allele can have.
        :rtype: float
        """
        return cls._max

    @classmethod
    def get_mid(cls):
        """
        :return: The middle value that the allele can have.
        """
        return (cls.get_min() + cls.get_max()) / 2

    def get(self, normalized=False):
        """
        :param normalized: Flag whether the value should be normalized.
        :type normalized: bool
        Returns the value of the allel.
        :return: The value of the allel.
        """
        if normalized:
            return (self.__value - self.__class__.get_min()) / (self.__class__.get_max() - self.__class__.get_min())
        return self.__value

    def mutate(self, method=MUT_GAUSS, heat=1):
        """
        Mutates the allele.
        :param method: The mutation method.
        :type method: function
        :param heat: The heat of the mutation.
        :type heat: float or int
        :return: The mutated allele.
        :rtype: Allele
        """
        method(self, heat)

    @classmethod
    def distance(cls, a, b, normalized=False):
        """
        Returns the distance between two alleles.
        :param a: The first allele.
        :type a: Allele
        :param b: The second allele.
        :type b: Allele
        :param normalized: Whether the distance should be normalized.
        :type normalized: bool
        :return: The distance between the two alleles.
        :rtype: float
        """
        if normalized:
            return abs(a.get(normalized=True) - b.get(normalized=True))
        else:
            return abs(a.get() - b.get())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__value == other.get()

    def __lt__(self, other):
        return self.__value < other.get()

    def __gt__(self, other):
        return self.__value > other.get()

    def __abs__(self):
        return abs(self.__value)

    def __add__(self, other):
        return self.__class__(self.__value + other.get())

    def __sub__(self, other):
        return self.__class__(self.__value - other.get())

    def __mul__(self, other):
        return self.__class__(self.__value * other.get())

    def __truediv__(self, other):
        return self.__class__(self.__value / other.get())

    def __floordiv__(self, other):
        return self.__class__(self.__value // other.get())

    def __copy__(self):
        return self.__class__(self.__value)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.__value}'

    def __str__(self):
        return str(self.__value)

    def __neg__(self):
        mid = self.get_mid()
        return self.__class__(2 * mid - self.__value)

    def __hash__(self):
        return hash(self.__value)
