#!/usr/bin/python3

import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import aiohttp


class GitHubAPIError(RuntimeError):
    """Raised when GitHub cannot provide a complete, valid response."""


###############################################################################
# Main Classes
###############################################################################

class Queries(object):
    """
    Class with functions to query the GitHub GraphQL (v4) API and the REST (v3)
    API. Also includes functions to dynamically generate GraphQL queries.
    """

    def __init__(self, username: str, access_token: str,
                 session: aiohttp.ClientSession, max_connections: int = 10,
                 max_attempts: int = 4, retry_base: float = 1.0,
                 stats_max_attempts: int = 60,
                 stats_retry_delay: float = 2.0):
        self.username = username
        self.access_token = access_token
        self.session = session
        self.semaphore = asyncio.Semaphore(max_connections)
        self.max_attempts = max(1, max_attempts)
        self.retry_base = max(0, retry_base)
        self.stats_max_attempts = max(1, stats_max_attempts)
        self.stats_retry_delay = max(0, stats_retry_delay)

    @staticmethod
    def _request_error(operation: str, status: int,
                       request_id: Optional[str] = None) -> GitHubAPIError:
        request_suffix = (f" (GitHub request {request_id})"
                          if request_id else "")
        return GitHubAPIError(
            f"{operation} failed with HTTP {status}{request_suffix}"
        )

    async def _request_json(
            self, method: str, url: str, operation: str,
            attempts: int, base_delay: float,
            retry_statuses: Set[int],
            validator: Optional[Callable[[Any], Optional[str]]] = None,
            exponential_backoff: bool = True,
            **kwargs: Any) -> Any:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/vnd.github+json",
        }
        last_error = GitHubAPIError(f"{operation} failed")

        for attempt in range(attempts):
            try:
                async with self.semaphore:
                    async with self.session.request(
                            method, url, headers=headers, **kwargs) as response:
                        status = response.status
                        request_id = response.headers.get("X-GitHub-Request-Id")

                        if status in retry_statuses:
                            last_error = self._request_error(
                                operation, status, request_id
                            )
                        elif not 200 <= status < 300:
                            raise self._request_error(
                                operation, status, request_id
                            )
                        else:
                            try:
                                result = await response.json(content_type=None)
                            except (ValueError, aiohttp.ClientError):
                                last_error = GitHubAPIError(
                                    f"{operation} returned invalid JSON"
                                    + (f" (GitHub request {request_id})"
                                       if request_id else "")
                                )
                            else:
                                validation_error = (validator(result)
                                                    if validator else None)
                                if validation_error is None:
                                    return result
                                last_error = GitHubAPIError(
                                    f"{operation} {validation_error}"
                                    + (f" (GitHub request {request_id})"
                                       if request_id else "")
                                )
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                last_error = GitHubAPIError(
                    f"{operation} failed with {type(exc).__name__}"
                )

            if attempt + 1 == attempts:
                raise last_error
            delay = base_delay * (2 ** attempt if exponential_backoff else 1)
            if exponential_backoff or attempt == 0 or (attempt + 1) % 10 == 0:
                print(
                    f"{last_error}; retrying {operation} after attempt "
                    f"{attempt + 1} in {delay:.1f}s"
                )
            await asyncio.sleep(delay)

        raise last_error

    async def query(self, generated_query: str) -> Dict:
        """
        Make a request to the GraphQL API using the authentication token from
        the environment
        :param generated_query: string query to be sent to the API
        :return: decoded GraphQL JSON output
        """
        def validate(result: Any) -> Optional[str]:
            if not isinstance(result, dict):
                return "returned an unexpected payload"
            if result.get("errors"):
                return "returned GraphQL errors"
            if not isinstance(result.get("data"), dict):
                return "returned no data"
            return None

        return await self._request_json(
            "POST",
            "https://api.github.com/graphql",
            "GraphQL query",
            self.max_attempts,
            self.retry_base,
            {408, 429, 500, 502, 503, 504},
            validator=validate,
            json={"query": generated_query},
        )

    async def query_rest(self, path: str,
                         params: Optional[Dict] = None) -> Any:
        """
        Make a request to the REST API
        :param path: API path to query
        :param params: Query parameters to be passed to the API
        :return: deserialized REST JSON output
        """

        normalized_path = path.lstrip("/")
        is_repository_stats = normalized_path.endswith("/stats/contributors")
        attempts = (self.stats_max_attempts if is_repository_stats
                    else self.max_attempts)
        delay = (self.stats_retry_delay if is_repository_stats
                 else self.retry_base)
        retry_statuses = {408, 429, 500, 502, 503, 504}
        if is_repository_stats:
            retry_statuses.add(202)

        return await self._request_json(
            "GET",
            f"https://api.github.com/{normalized_path}",
            ("repository contributor statistics" if is_repository_stats
             else "REST query"),
            attempts,
            delay,
            retry_statuses,
            exponential_backoff=not is_repository_stats,
            params=params or {},
        )

    @staticmethod
    def owned_repos(cursor: Optional[str] = None) -> str:
        """
        :return: GraphQL query with overview of user repositories
        """
        return f"""{{
  viewer {{
    login,
    name,
    repositories(
        first: 10,
        orderBy: {{
            field: UPDATED_AT,
            direction: DESC
        }},
        isFork: false,
        after: {json.dumps(cursor)}
    ) {{
      pageInfo {{
        hasNextPage
        endCursor
      }}
      nodes {{
        nameWithOwner
        stargazerCount
        forkCount
        languages(first: 10, orderBy: {{field: SIZE, direction: DESC}}) {{
          edges {{
            size
            node {{
              name
              color
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

    @staticmethod
    def contributed_repos(cursor: Optional[str] = None) -> str:
        """Return a paginated query for repositories the user contributed to."""
        return f"""{{
  viewer {{
    login,
    name,
    repositoriesContributedTo(
        first: 10,
        includeUserRepositories: false,
        orderBy: {{
            field: UPDATED_AT,
            direction: DESC
        }},
        contributionTypes: [
            COMMIT,
            PULL_REQUEST,
            REPOSITORY,
            PULL_REQUEST_REVIEW
        ]
        after: {json.dumps(cursor)}
    ) {{
      pageInfo {{
        hasNextPage
        endCursor
      }}
      nodes {{
        nameWithOwner
        stargazerCount
        forkCount
        languages(first: 10, orderBy: {{field: SIZE, direction: DESC}}) {{
          edges {{
            size
            node {{
              name
              color
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

    @staticmethod
    def contrib_years() -> str:
        """
        :return: GraphQL query to get all years the user has been a contributor
        """
        return """
query {
  viewer {
    contributionsCollection {
      contributionYears
    }
  }
}
"""

    @staticmethod
    def contribs_by_year(year: str) -> str:
        """
        :param year: year to query for
        :return: portion of a GraphQL query with desired info for a given year
        """
        return f"""
    year{year}: contributionsCollection(
        from: "{year}-01-01T00:00:00Z",
        to: "{int(year) + 1}-01-01T00:00:00Z"
    ) {{
      contributionCalendar {{
        totalContributions
      }}
    }}
"""

    @classmethod
    def all_contribs(cls, years: List[str]) -> str:
        """
        :param years: list of years to get contributions for
        :return: query to retrieve contribution information for all user years
        """
        by_years = "\n".join(map(cls.contribs_by_year, years))
        return f"""
query {{
  viewer {{
    {by_years}
  }}
}}
"""


class Stats(object):
    """
    Retrieve and store statistics about GitHub usage.
    """
    def __init__(self, username: str, access_token: str,
                 session: aiohttp.ClientSession,
                 exclude_repos: Optional[Set] = None,
                 exclude_langs: Optional[Set] = None,
                 consider_forked_repos: bool = False):
        self.username = username
        self._exclude_repos = set() if exclude_repos is None else exclude_repos
        self._exclude_langs = set() if exclude_langs is None else exclude_langs
        self._consider_forked_repos = consider_forked_repos
        self.queries = Queries(username, access_token, session)

        self._name = None
        self._stargazers = None
        self._forks = None
        self._total_contributions = None
        self._languages = None
        self._repos = None
        self._ignored_repos = None
        self._lines_changed = None
        self._views = None
        self._stats_task = None

    async def to_str(self) -> str:
        """
        :return: summary of all available statistics
        """
        languages = await self.languages_proportional
        formatted_languages = "\n  - ".join(
            [f"{k}: {v:0.4f}%" for k, v in languages.items()]
        )
        lines_changed = await self.lines_changed
        return f"""Name: {await self.name}
Stargazers: {await self.stargazers:,}
Forks: {await self.forks:,}
All-time contributions: {await self.total_contributions:,}
Repositories with contributions: {len(await self.all_repos)}
Lines of code added: {lines_changed[0]:,}
Lines of code deleted: {lines_changed[1]:,}
Lines of code changed: {lines_changed[0] + lines_changed[1]:,}
Project page views: {await self.views:,}
Languages:
  - {formatted_languages}"""

    async def _repository_pages(
            self, connection_name: str,
            query_builder: Callable[[Optional[str]], str]
    ) -> Tuple[List[Dict], str]:
        repos = []
        cursor = None
        seen_cursors = set()
        display_name = None

        while True:
            raw_results = await self.queries.query(query_builder(cursor))
            viewer = raw_results.get("data", {}).get("viewer")
            if not isinstance(viewer, dict):
                raise GitHubAPIError(
                    "GraphQL query returned incomplete viewer data"
                )

            if display_name is None:
                display_name = viewer.get("name") or viewer.get("login")
            connection = viewer.get(connection_name)
            if not isinstance(connection, dict):
                raise GitHubAPIError(
                    "GraphQL query returned incomplete repository data"
                )
            nodes = connection.get("nodes")
            page_info = connection.get("pageInfo")
            if (not isinstance(nodes, list)
                    or not isinstance(page_info, dict)
                    or any(not isinstance(repo, dict) for repo in nodes)):
                raise GitHubAPIError(
                    "GraphQL query returned malformed repository data"
                )
            repos.extend(nodes)

            if not page_info.get("hasNextPage", False):
                break
            cursor = page_info.get("endCursor")
            if not isinstance(cursor, str) or cursor in seen_cursors:
                raise GitHubAPIError(
                    "GraphQL repository pagination returned an invalid cursor"
                )
            seen_cursors.add(cursor)

        return repos, display_name or "No Name"

    async def _load_stats(self) -> None:
        """Fetch complete repository data, then publish it to this instance."""
        (owned_result, contributed_result) = await asyncio.gather(
            self._repository_pages("repositories", Queries.owned_repos),
            self._repository_pages(
                "repositoriesContributedTo", Queries.contributed_repos
            ),
        )
        owned_repos, owned_name = owned_result
        contributed_repos, contributed_name = contributed_result

        repos = set()
        ignored_repos = set()
        languages: Dict[str, Dict] = {}
        stargazers = 0
        forks = 0

        selected_repos = list(owned_repos)
        if self._consider_forked_repos:
            selected_repos.extend(contributed_repos)

        for repo in selected_repos:
            repo_name = repo.get("nameWithOwner")
            if repo_name in repos or repo_name in self._exclude_repos:
                continue

            repos.add(repo_name)
            stargazers += repo.get("stargazerCount", 0)
            forks += repo.get("forkCount", 0)

            for language in repo.get("languages", {}).get("edges", []):
                node = language.get("node", {})
                language_name = node.get("name", "Other")
                if language_name in self._exclude_langs:
                    continue
                if language_name not in languages:
                    languages[language_name] = {
                        "size": 0,
                        "occurrences": 0,
                        "color": node.get("color"),
                    }
                languages[language_name]["size"] += language.get("size", 0)
                languages[language_name]["occurrences"] += 1

        if not self._consider_forked_repos:
            for repo in contributed_repos:
                repo_name = repo.get("nameWithOwner")
                if (repo_name not in repos
                        and repo_name not in self._exclude_repos):
                    ignored_repos.add(repo_name)

        # TODO: Improve languages to scale by number of contributions to
        #       specific filetypes
        languages_total = sum(
            language["size"] for language in languages.values()
        )
        if languages_total > 0:
            for language in languages.values():
                language["prop"] = 100 * language["size"] / languages_total

        self._name = owned_name or contributed_name
        self._stargazers = stargazers
        self._forks = forks
        self._languages = languages
        self._repos = repos
        self._ignored_repos = ignored_repos

    async def get_stats(self) -> None:
        """Load repository statistics once and share the result across callers."""
        if self._stats_task is None:
            self._stats_task = asyncio.ensure_future(self._load_stats())
        await asyncio.shield(self._stats_task)

    @property
    async def name(self) -> str:
        """
        :return: GitHub user's name (e.g., Jacob Strieb)
        """
        if self._name is not None:
            return self._name
        await self.get_stats()
        assert(self._name is not None)
        return self._name

    @property
    async def stargazers(self) -> int:
        """
        :return: total number of stargazers on user's repos
        """
        if self._stargazers is not None:
            return self._stargazers
        await self.get_stats()
        assert(self._stargazers is not None)
        return self._stargazers

    @property
    async def forks(self) -> int:
        """
        :return: total number of forks on user's repos
        """
        if self._forks is not None:
            return self._forks
        await self.get_stats()
        assert(self._forks is not None)
        return self._forks

    @property
    async def languages(self) -> Dict:
        """
        :return: summary of languages used by the user
        """
        if self._languages is not None:
            return self._languages
        await self.get_stats()
        assert(self._languages is not None)
        return self._languages

    @property
    async def languages_proportional(self) -> Dict:
        """
        :return: summary of languages used by the user, with proportional usage
        """
        if self._languages is None:
            await self.get_stats()
            assert(self._languages is not None)

        return {k: v.get("prop", 0) for (k, v) in self._languages.items()}

    @property
    async def repos(self) -> List[str]:
        """
        :return: list of names of user's repos
        """
        if self._repos is not None:
            return self._repos
        await self.get_stats()
        assert(self._repos is not None)
        return self._repos
    
    @property
    async def all_repos(self) -> List[str]:
        """
        :return: list of names of user's repos with contributed repos included
                irrespective of whether the ignore flag is set or not
        """
        if self._repos is not None and self._ignored_repos is not None:
            return self._repos | self._ignored_repos
        await self.get_stats()
        assert(self._repos is not None)
        assert(self._ignored_repos is not None)
        return self._repos | self._ignored_repos

    @property
    async def total_contributions(self) -> int:
        """
        :return: count of user's total contributions as defined by GitHub
        """
        if self._total_contributions is not None:
            return self._total_contributions

        self._total_contributions = 0
        years = (await self.queries.query(Queries.contrib_years())) \
            .get("data", {}) \
            .get("viewer", {}) \
            .get("contributionsCollection", {}) \
            .get("contributionYears", [])
        by_year = (await self.queries.query(Queries.all_contribs(years))) \
            .get("data", {}) \
            .get("viewer", {}).values()
        for year in by_year:
            self._total_contributions += year \
                .get("contributionCalendar", {}) \
                .get("totalContributions", 0)
        return self._total_contributions

    @property
    async def lines_changed(self) -> Tuple[int, int]:
        """
        :return: count of total lines added, removed, or modified by the user
        """
        if self._lines_changed is not None:
            return self._lines_changed
        additions = 0
        deletions = 0
        for repo in await self.all_repos:
            result = await self.queries.query_rest(
                f"/repos/{repo}/stats/contributors"
            )
            if not isinstance(result, list):
                raise GitHubAPIError(
                    "REST query returned malformed contributor statistics"
                )
            for author_obj in result:
                # Handle malformed response from the API by skipping this repo
                if (not isinstance(author_obj, dict)
                        or not isinstance(author_obj.get("author", {}), dict)):
                    continue
                author = author_obj.get("author", {}).get("login", "")
                if author != self.username:
                    continue

                for week in author_obj.get("weeks", []):
                    additions += week.get("a", 0)
                    deletions += week.get("d", 0)

        self._lines_changed = (additions, deletions)
        return self._lines_changed

    @property
    async def views(self) -> int:
        """
        Note: only returns views for the last 14 days (as-per GitHub API)
        :return: total number of page views the user's projects have received
        """
        if self._views is not None:
            return self._views

        total = 0
        for repo in await self.repos:
            result = await self.queries.query_rest(
                f"/repos/{repo}/traffic/views"
            )
            for view in result.get("views", []):
                total += view.get("count", 0)

        self._views = total
        return total


###############################################################################
# Main Function
###############################################################################

async def main() -> None:
    """
    Used mostly for testing; this module is not usually run standalone
    """
    access_token = os.getenv("ACCESS_TOKEN")
    user = os.getenv("GITHUB_ACTOR")
    async with aiohttp.ClientSession() as session:
        s = Stats(user, access_token, session)
        print(await s.to_str())


if __name__ == "__main__":
    asyncio.run(main())
