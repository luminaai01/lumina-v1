from typing import Optional, Type

from asyncer import asyncify

from lumina.actions.base_action import AsyncActionMixin, BaseAction, tool_api
from lumina.actions.parser import BaseParser, JsonParser
from lumina.schema import ActionReturn, ActionStatusCode


class ArxivSearch(BaseAction):
    """Search information from Arxiv.org. \
Useful for when you need to answer questions about Physics, Mathematics, \
Computer Science, Quantitative Biology, Quantitative Finance, Statistics, \
Electrical Engineering, and Economics from scientific articles on arxiv.org.
    """

    def __init__(
        self,
        top_k_results: int = 3,
        max_query_len: int = 300,
        doc_content_chars_max: int = 1500,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)
        self.top_k_results = top_k_results
        self.max_query_len = max_query_len
        self.doc_content_chars_max = doc_content_chars_max

    @tool_api(explode_return=True)
    def get_arxiv_article_information(self, query: str) -> dict:
        """Run Arxiv search and get the article meta information.

        Args:
            query (:class:`str`): the content of search query

        Returns:
            :class:`dict`: article information
                * content (str): a list of 3 arxiv search papers
        """
        import arxiv

        try:
            results = arxiv.Search(  # type: ignore
                query[: self.max_query_len], max_results=self.top_k_results
            ).results()
        except Exception as exc:
            return ActionReturn(errmsg=f'Arxiv exception: {exc}', state=ActionStatusCode.HTTP_ERROR)
        docs = [
            f'Published: {result.updated.date()}\nTitle: {result.title}\n'
            f'Authors: {", ".join(a.name for a in result.authors)}\n'
            f'Summary: {result.summary[:self.doc_content_chars_max]}'
            for result in results
        ]
        if docs:
            return {'content': '\n\n'.join(docs)}
        return {'content': 'No good Arxiv Result was found'}


class AsyncArxivSearch(AsyncActionMixin, ArxivSearch):
    """Search information from Arxiv.org. \
Useful for when you need to answer questions about Physics, Mathematics, \
Computer Science, Quantitative Biology, Quantitative Finance, Statistics, \
Electrical Engineering, and Economics from scientific articles on arxiv.org.
    """

    @tool_api(explode_return=True)
    @asyncify
    def get_arxiv_article_information(self, query: str) -> dict:
        """Run Arxiv search and get the article meta information.

        Args:
            query (:class:`str`): the content of search query

        Returns:
            :class:`dict`: article information
                * content (str): a list of 3 arxiv search papers
        """
        return super().get_arxiv_article_information(query)
