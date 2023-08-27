class Error(Exception):
    def __init__(self, message):
        self.message = message


class BadlySpecifiedTemplateError(Error):
    pass

class EmptyStackException(Exception):
  pass

class ReachedEndOfNodeException(Exception):
    def __init__(self, matcher_outer_node, matcher_inner_node, remaining_string):
        self.matcher_outer_node = matcher_outer_node
        self.matcher_inner_node = matcher_inner_node
        self.remaining_string = remaining_string
    def __str__(self):
        return (f'(outer matcher: {repr(self.matcher_outer_node)}, ineer_matcher {repr(self.matcher_inner_node)})')