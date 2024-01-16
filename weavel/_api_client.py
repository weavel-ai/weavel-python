import os
from typing import Dict

import requests

import httpx
from rich import print

from weavel.utils.crypto import decrypt_message


class APIClient:
    """
    A class to represent an API request client.

    ...

    Methods
    -------
    get_headers():
        Generates headers for the API request.
    execute(method="GET", params=None, data=None, json=None, **kwargs):
        Executes the API request.
    """

    @classmethod
    def _get_headers(cls, api_key) -> Dict:
        """
        Reads, decrypts the api_key, and returns headers for API request.

        Returns
        -------
        dict
            a dictionary containing the Authorization header
        """
        headers = {"Authorization": f"Bearer {api_key}"}
        return headers

    @classmethod
    def execute(
        cls,
        api_key,
        endpoint: str,
        path: str,
        method="GET",
        params: Dict = None,
        data: Dict = None,
        json: Dict = None,
        ignore_auth_error: bool = False,
        **kwargs,
    ) -> requests.Response:
        """
        Executes the API request with the decrypted API key in the headers.

        Parameters
        ----------
        method : str, optional
            The HTTP method of the request (default is "GET")
        params : dict, optional
            The URL parameters to be sent with the request
        data : dict, optional
            The request body to be sent with the request
        json : dict, optional
            The JSON-encoded request body to be sent with the request
        ignore_auth_error: bool, optional
            Whether to ignore authentication errors (default is False)
        **kwargs : dict
            Additional arguments to pass to the requests.request function

        Returns
        -------
        requests.Response
            The response object returned by the requests library
        """
        url = f"{endpoint}{path}"
        headers = cls._get_headers(api_key)
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                data=data,
                json=json,
                **kwargs,
            )
            if not response:
                print(f"[red]Error: {response}[/red]")
                exit()
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                if not ignore_auth_error:
                    print(
                        "[red]Authentication failed. Check out user [violet][bold]WEAVEL_API_KEY[/bold][/violet].[/red]"
                    )
                    exit()
            else:
                print(f"[red]Error: {response}[/red]")
                exit()
        except requests.exceptions.ConnectionError:
            print("[red]Could not connect to the WEAVEL API.[/red]")
        except requests.exceptions.Timeout:
            print("[red]The request timed out.[/red]")


class AsyncAPIClient:
    """
    A class to represent an Async API request client.
    Used in Deployment stage.

    ...

    Methods
    -------
    get_headers():
        Generates headers for the API request.
    execute(method="GET", params=None, data=None, json=None, **kwargs):
        Executes the API request.
    """
        

    @classmethod
    async def _get_headers(cls, api_key: str) -> Dict:
        """
        Reads, decrypts the api_key, and returns headers for API request.

        Returns
        -------
        dict
            a dictionary containing the Authorization header
        """
        
        headers = {"Authorization": f"Bearer {api_key}"}
        return headers

    @classmethod
    async def execute(
        cls,
        api_key,
        endpoint: str,
        path: str,
        method="GET",
        params: Dict = None,
        data: Dict = None,
        json: Dict = None,
        ignore_auth_error: bool = False,
        **kwargs,
    ) -> requests.Response:
        """
        Executes the API request with the decrypted API key in the headers.

        Parameters
        ----------
        method : str, optional
            The HTTP method of the request (default is "GET")
        params : dict, optional
            The URL parameters to be sent with the request
        data : dict, optional
            The request body to be sent with the request
        json : dict, optional
            The JSON-encoded request body to be sent with the request
        ignore_auth_error: bool, optional
            Whether to ignore authentication errors (default is False)
        **kwargs : dict
            Additional arguments to pass to the requests.request function

        Returns
        -------
        requests.Response
            The response object returned by the requests library
        """
        url = f"{endpoint}{path}"
        headers = await cls._get_headers(api_key)
        try:
            async with httpx.AsyncClient(http2=True) as _client:
                response = await _client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    data=data,
                    json=json,
                    **kwargs,
                )
            if not response:
                print(f"[red]Error: {response}[/red]")
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                if not ignore_auth_error:
                    print("[red]Authentication failed.[/red]")
            else:
                print(f"[red]Error: {response}[/red]")

            return response
        except requests.exceptions.ConnectionError:
            print("[red]Could not connect to the WEAVEL API.[/red]")
        except requests.exceptions.Timeout:
            print("[red]The request timed out.[/red]")
        except Exception as exception:
            print(f"[red]Error: {exception}[/red]")
