import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, call, patch

import generate_images
from github_stats import GitHubAPIError, Queries, Stats


class FakeResponse:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status = status
        self._payload = payload
        self._text = text
        self.headers = headers or {}

    async def __aenter__(self):
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self, content_type=None):
        if self._payload is None:
            raise json.JSONDecodeError("Expecting value", self._text, 0)
        return self._payload

    async def text(self):
        return self._text


class NoContentResponse(FakeResponse):
    def __init__(self):
        super().__init__(204)

    async def json(self, content_type=None):
        raise AssertionError("204 responses must not be decoded as JSON")


class QueueSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class RepositorySession:
    def __init__(self):
        self.queries = []

    def request(self, method, url, **kwargs):
        query = kwargs["json"]["query"]
        self.queries.append(query)

        if "repositoriesContributedTo(" in query:
            connection = {
                "nodes": [],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
            viewer = {
                "name": "Example User",
                "login": "example",
                "repositoriesContributedTo": connection,
            }
        else:
            connection = {
                "nodes": [
                    {
                        "nameWithOwner": "example/repo",
                        "stargazerCount": 1,
                        "forkCount": 2,
                        "languages": {
                            "edges": [
                                {
                                    "size": 10,
                                    "node": {
                                        "name": "Python",
                                        "color": "#3572A5",
                                    },
                                }
                            ]
                        },
                    }
                ],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
            viewer = {
                "name": "Example User",
                "login": "example",
                "repositories": connection,
            }

        return FakeResponse(200, {"data": {"viewer": viewer}})


class PaginatedRepositorySession:
    def __init__(self):
        self.queries = []

    @staticmethod
    def repo(name):
        return {
            "nameWithOwner": name,
            "stargazerCount": 1,
            "forkCount": 1,
            "languages": {"edges": []},
        }

    def request(self, method, url, **kwargs):
        query = kwargs["json"]["query"]
        self.queries.append(query)

        if "repositoriesContributedTo(" in query:
            connection_name = "repositoriesContributedTo"
            nodes = [self.repo("someone/contributed")]
            page_info = {"hasNextPage": False, "endCursor": None}
        elif 'after: "owned-cursor"' in query:
            connection_name = "repositories"
            nodes = [self.repo("example/owned-two")]
            page_info = {"hasNextPage": False, "endCursor": None}
        else:
            connection_name = "repositories"
            nodes = [self.repo("example/owned-one")]
            page_info = {
                "hasNextPage": True,
                "endCursor": "owned-cursor",
            }

        viewer = {
            "name": "Example User",
            "login": "example",
            connection_name: {"nodes": nodes, "pageInfo": page_info},
        }
        return FakeResponse(200, {"data": {"viewer": viewer}})


class QueriesTest(unittest.IsolatedAsyncioTestCase):
    async def test_rest_204_returns_no_content_without_decoding_json(self):
        session = QueueSession([NoContentResponse()])
        queries = Queries(
            "example", "token", session, stats_max_attempts=1
        )

        result = await queries.query_rest(
            "/repos/example/repo/stats/contributors"
        )

        self.assertIsNone(result)

    async def test_retries_transient_non_json_response(self):
        session = QueueSession(
            [
                FakeResponse(
                    502,
                    text="<html>Bad Gateway</html>",
                    headers={"X-GitHub-Request-Id": "request-1"},
                ),
                FakeResponse(200, {"data": {"viewer": {"login": "example"}}}),
            ]
        )
        queries = Queries(
            "example", "token", session, max_attempts=2, retry_base=0
        )

        result = await queries.query("query { viewer { login } }")

        self.assertEqual("example", result["data"]["viewer"]["login"])
        self.assertEqual(2, len(session.calls))

    async def test_non_json_response_exhaustion_raises_api_error(self):
        session = QueueSession(
            [
                FakeResponse(502, text="<html>Bad Gateway</html>"),
                FakeResponse(502, text="<html>Bad Gateway</html>"),
            ]
        )
        queries = Queries(
            "example", "token", session, max_attempts=2, retry_base=0
        )

        with self.assertRaisesRegex(GitHubAPIError, "HTTP 502"):
            await queries.query("query { viewer { login } }")

    async def test_rest_202_exhaustion_is_not_treated_as_empty_data(self):
        session = QueueSession([FakeResponse(202), FakeResponse(202)])
        queries = Queries(
            "example",
            "token",
            session,
            stats_max_attempts=2,
            stats_retry_delay=0,
        )

        with self.assertRaisesRegex(GitHubAPIError, "HTTP 202"):
            await queries.query_rest("/repos/example/repo/stats/contributors")


class StatsTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_contributor_stats_is_zero_without_user_commits(self):
        stats = Stats("example", "token", QueueSession([]))
        stats._repos = {"example/repo"}
        stats._ignored_repos = set()
        stats.queries.query_rest = AsyncMock(side_effect=[None, []])

        self.assertEqual((0, 0), await stats.lines_changed)
        stats.queries.query_rest.assert_has_awaits(
            [
                call("/repos/example/repo/stats/contributors"),
                call(
                    "/repos/example/repo/commits",
                    params={"author": "example", "per_page": 1},
                ),
            ]
        )

    async def test_no_contributor_stats_fails_when_user_has_commits(self):
        stats = Stats("example", "token", QueueSession([]))
        stats._repos = {"example/repo"}
        stats._ignored_repos = set()
        stats.queries.query_rest = AsyncMock(
            side_effect=[None, [{"sha": "commit"}]]
        )

        with self.assertRaisesRegex(GitHubAPIError, "example/repo"):
            await stats.lines_changed

    async def test_concurrent_initial_reads_share_one_paginated_load(self):
        session = RepositorySession()
        stats = Stats("example", "token", session)

        languages, name = await asyncio.gather(stats.languages, stats.name)

        self.assertEqual("Example User", name)
        self.assertEqual({"Python"}, set(languages))
        self.assertEqual(1, await stats.stargazers)
        self.assertEqual(2, len(session.queries))
        self.assertEqual(
            1,
            sum("repositoriesContributedTo(" in query for query in session.queries),
        )
        self.assertEqual(
            1,
            sum(
                "repositories(" in query
                and "repositoriesContributedTo(" not in query
                for query in session.queries
            ),
        )
        self.assertTrue(all("first: 10" in query for query in session.queries))

    async def test_repository_connections_paginate_independently(self):
        session = PaginatedRepositorySession()
        stats = Stats("example", "token", session)

        repos = await stats.all_repos

        self.assertEqual(
            {
                "example/owned-one",
                "example/owned-two",
                "someone/contributed",
            },
            repos,
        )
        self.assertEqual(3, len(session.queries))
        self.assertEqual(
            2,
            sum(
                "repositories(" in query
                and "repositoriesContributedTo(" not in query
                for query in session.queries
            ),
        )
        self.assertEqual(
            1,
            sum("repositoriesContributedTo(" in query for query in session.queries),
        )


class FakeClientSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class IncompleteStats:
    async def get_stats(self):
        raise GitHubAPIError("statistics are incomplete")

    @property
    async def total_contributions(self):
        return 0

    @property
    async def lines_changed(self):
        return (0, 0)

    @property
    async def views(self):
        return 0


class GenerateImagesTest(unittest.IsolatedAsyncioTestCase):
    async def test_incomplete_statistics_do_not_write_images(self):
        generate_languages = AsyncMock()
        generate_overview = AsyncMock()
        with patch.dict(
                os.environ,
                {
                    "ACCESS_TOKEN": "token",
                    "GITHUB_ACTOR": "example",
                    "COUNT_STATS_FROM_FORKS": "",
                },
                clear=True), patch.object(
                    generate_images.aiohttp,
                    "ClientSession",
                    return_value=FakeClientSession()), patch.object(
                        generate_images,
                        "Stats",
                        return_value=IncompleteStats()), patch.object(
                            generate_images,
                            "generate_languages",
                            generate_languages), patch.object(
                                generate_images,
                                "generate_overview",
                                generate_overview):
            with self.assertRaisesRegex(GitHubAPIError, "incomplete"):
                await generate_images.main()

        generate_languages.assert_not_awaited()
        generate_overview.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
