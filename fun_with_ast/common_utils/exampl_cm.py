from contextlib import contextmanager


class B():
    def __init__(self):
        print('This is B init')

    def print_context(self):
        print(swapped_string)


class A():
    def __init__(self):
        self._B = B()
        print('this is A init')

    def call_B_to_print_context(self):
        self._B.print_context()

class C():
    def __init__(self):
        print('This is C init')

    def print_context(self):
        #with example_cm('This Is A Test For Context manager') as swapped_string:
            a = A()
            a.call_B_to_print_context()


@contextmanager
def example_cm(string_input):
    print('Setup logic')

    swapped = string_input.swapcase()
    try:
        yield swapped
    except ValueError as e:
        print('An error occurred...')
    finally:
        print('Teardown logic')
        del swapped

    print('End of context manager\n')



if __name__ == '__main__':
    with example_cm('This Is A Test For Context manager') as swapped_string:
        c = C()
        c.print_context()

    #a = A()
    #a.call_B_to_print_context()
