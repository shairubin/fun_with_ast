

from contextlib import contextmanager

@contextmanager
def MatchCM(full_string):
   print('Setup logic')
   try:
       yield full_string
   except Exception as e:
       print('An error occurred...')
       raise
   finally:
       pass




