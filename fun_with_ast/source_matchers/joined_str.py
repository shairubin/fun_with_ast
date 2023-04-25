from list_placeholder_source_match import ListFieldPlaceholder
from fun_with_ast.source_matchers.str_source_match import StrSourceMatcher
from fun_with_ast.string_part_placeholder import JoinedStringPartPlaceholder
from fun_with_ast.source_matcher_source_match import MatchPlaceholder


class JoinedStrSourceMatcher(StrSourceMatcher):
    def __init__(self, node, starting_parens=None):
        super(JoinedStrSourceMatcher, self).__init__(node, starting_parens)
        self.value_placeholder = ListFieldPlaceholder('values')

    def _get_original_string(self):
        self.original_s = ''

    def _part_place_holder(self):
        return JoinedStringPartPlaceholder()

    def Match(self, string):
        part = self._part_place_holder()
        remaining_string = MatchPlaceholder(string, None, part)
        self.quote_parts.append(part)

    def _handle_multipart(self, remaining_string):
        pass