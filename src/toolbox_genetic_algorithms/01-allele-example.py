"""
This examples show how to use the Allele class to create a new allele type and use the get and set methods with and
without normalization.
"""
import evolution as evo


class RollRadius(evo.Allele):
    _min = 0.2
    _max = 0.5


if __name__ == '__main__':
    # instantiate the allele with a value of 0.3
    radius = RollRadius(.3)

    # print the value
    print(f'radius: {radius.get()}')

    # print the normalized value
    print(f'normalized radius: {radius.get(normalized=True)}')

    # set the value to 0.5 by passing a normalized value of 1
    radius.set(1, normalized=True)
    print(f'radius: {radius.get()}')
    print(f'normalized radius: {radius.get(normalized=True)}')
