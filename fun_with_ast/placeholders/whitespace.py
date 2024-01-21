from fun_with_ast.placeholders.text import TextPlaceholder

class WSStartOfLinePlaceholder(TextPlaceholder):
    def __init__(self):
        super(WSStartOfLinePlaceholder, self).__init__(r'[ \t]*', default='', no_transform=True)
class WSEndOfLinePlaceholder(TextPlaceholder):
    def __init__(self):
        super(WSEndOfLinePlaceholder, self).__init__(r'[ \t]*(?=\n)', default='', no_transform=True)

class EOLPlaceholder(TextPlaceholder):
    def __init__(self):
        super(EOLPlaceholder, self).__init__(r'\n', default='', longest_match=False)

class EOLCommentMatcher(TextPlaceholder):
    def __init__(self):
        super(EOLCommentMatcher, self).__init__(r'([ \t]*)(#.*)$', default='', longest_match=False)
