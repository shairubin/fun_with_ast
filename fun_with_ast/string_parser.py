# #from fun_with_ast.source_match import GetStartParenMatcher
#
# from placeholder_source_match import Placeholder
# #from source_match import StripStartParens, BadlySpecifiedTemplateError, GetSource
# from text_placeholder_source_match import TextPlaceholder
#
#
# def GetSource(field, text=None, starting_parens=None, assume_no_indent=False):
#     """Gets the source corresponding with a given field.
#
#     If the node is not a string or a node with a .matcher function,
#     this will get the matcher for the node, attach the matcher, and
#     match the text provided. If no text is provided, it will rely on defaults.
#
#     Args:
#       field: {str|_ast.AST} The field we want the source from.
#       text: {str} The text to match if a matcher doesn't exist.
#       starting_parens: {[TextPlaceholder]} The list of parens that the field
#           starts with.
#       assume_no_indent: {bool} True if we can assume the node isn't indented.
#           Used for things like new nodes that aren't yet in a module.
#
#     Returns:
#       A string, representing the source code for the node.
#
#     Raises:
#       ValueError: When passing in a stmt node that has no string or module_node.
#           This is an error because we have no idea how much to indent it.
#     """
#     if field is None:
#         return ''
#     if starting_parens is None:
#         starting_parens = []
#     if isinstance(field, str):
#         return field
#     if isinstance(field, int):
#         return str(field)
#     if hasattr(field, 'matcher') and field.matcher:
#         return field.matcher.GetSource()
#     else:
#         field.matcher = GetMatcher(field, starting_parens)
#         if text:
#             field.matcher.Match(text)
#         # TODO: Fix this to work with lambdas
#         elif isinstance(field, _ast.stmt) and not assume_no_indent:
#             if not hasattr(field, 'module_node'):
#                 raise ValueError(
#                     'No text was provided, and we try to get source from node {} which'
#                     'is a statement, so it must have a .module_node field defined. '
#                     'To add this automatically, call ast_annotate.AddBasicAnnotations'
#                         .format(field))
#             FixSourceIndentation(field.module_node, field)
#
#         source_code = field.matcher.GetSource()
#         return source_code
#
#
# def GetStartParenMatcher():
#     return TextPlaceholder(r'[ \t]*\(\s*', '')
#
# def StripStartParens(string):
#     remaining_string = string
#     while remaining_string.startswith('('):
#         matcher = GetStartParenMatcher()
#         matched_text = matcher.Match(None, remaining_string)
#         remaining_string = remaining_string[len(matched_text):]
#     return remaining_string
#
# class StringParser(object):
#     """Class encapsulating parsing a string while matching placeholders."""
#
#     def __init__(self, string, elements, starting_parens=None):
#         if not starting_parens:
#             starting_parens = []
#         self.starting_parens = starting_parens
#         self.string = string
#         self.before_string = None
#         self.remaining_string = string
#         self.elements = elements
#         self.matched_substrings = []
#         self.Parse()
#
#     def _ProcessSubstring(self, substring):
#         """Process a substring, validating its state and calculating remaining."""
#         if not substring:
#             return
#         stripped_substring = StripStartParens(substring)
#         stripped_remaining = StripStartParens(self.remaining_string)
#         if not stripped_remaining.startswith(stripped_substring):
#             raise BadlySpecifiedTemplateError(
#                 'string "{}" should be in string "{}"'
#                     .format(stripped_substring, stripped_remaining))
#         self.remaining_string = self.remaining_string.split(
#             stripped_substring, 1)[1]
#
#     def _MatchTextPlaceholder(self, element):
#         if self.remaining_string == self.string:
#             element.SetStartingParens(self.starting_parens)
#         matched_text = element.Match(None, self.remaining_string)
#         self._ProcessSubstring(matched_text)
#         self.matched_substrings.append(matched_text)
#
#     def _MatchNode(self, node):
#         starting_parens = []
#         if self.remaining_string == self.string:
#             starting_parens = self.starting_parens
#         node_src = GetSource(node, self.remaining_string, starting_parens)
#         self._ProcessSubstring(node_src)
#         self.matched_substrings.append(node_src)
#
#     def GetMatchedText(self):
#         return ''.join(self.matched_substrings)
#
#     def Parse(self):
#         """Parses the string, handling nodes and TextPlaceholders."""
#         for element in self.elements:
#             if isinstance(element, Placeholder):
#                 self._MatchTextPlaceholder(element)
#             else:
#                 self._MatchNode(element)
