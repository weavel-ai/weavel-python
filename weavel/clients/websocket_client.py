import asyncio
from functools import wraps
import json
import datetime
import os
import re
import textwrap
import time  # Add this import

from uuid import UUID, uuid4
from typing import Callable, Dict, Any, Optional, AsyncGenerator, List
from dotenv import load_dotenv
from collections import defaultdict
from asyncio import Queue, Semaphore

from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from readerwriterlock import rwlock
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from weavel.utils.logging import logger
from weavel._constants import ENDPOINT_URL
from weavel.types.websocket import WsLocalTask, WsServerTask


load_dotenv()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            aware_datetime = obj.replace(tzinfo=datetime.timezone.utc)
            return aware_datetime.isoformat()  # This will include timezone information
        return super().default(obj)


def websocket_handler(message_type: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, data: Dict[str, Any]):
            try:
                result = await func(self, data)
                if result is not None:
                    response = {
                        "type": f"{message_type}_response",
                        "correlation_id": data.get("correlation_id"),
                        "data": result,
                    }
                    logger.debug(f"Sending response for {message_type}: {response}")
                    await self.ws_client.send_message(response)
            except Exception as e:
                logger.exception(f"Error in {func.__name__}")

        wrapper.message_type = message_type
        return wrapper

    return decorator


class WebsocketClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WebsocketClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
    ):
        if not hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.api_key = api_key
        protocol = "ws" if os.getenv("WEAVEL_TESTMODE") == "true" else "wss"
        base_url = base_url or ENDPOINT_URL
        domain = base_url.split("://")[1]
        self.endpoint = f"{protocol}://{domain}/public/v2/ws/open"
        self.ws: Optional[WebSocketClientProtocol] = None
        self.message_handlers: Dict[str, Callable] = {}
        self.heartbeat_task = None
        self.message_handler_task = None

        # Correlation Management
        self._pending_requests: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._lock = asyncio.Lock()  # To ensure thread-safe operations
        self._timeout_reset_events: Dict[str, asyncio.Event] = defaultdict(
            asyncio.Event
        )

        self._semaphore = Semaphore(100)  # Limit to 100 concurrent tasks

    async def __aenter__(self):
        await self.connect_to_gateway()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_connection()

    @retry(
        stop=stop_after_attempt(12 * 24),
        wait=wait_fixed(5 * 60),
        retry=retry_if_exception_type(
            (ConnectionClosedError, ConnectionClosedOK, TimeoutError)
        ),
    )
    async def connect_to_gateway(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            self.ws = await connect(
                self.endpoint,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=60,
                close_timeout=60 * 5,
            )
            logger.info("WebSocket connection established")
            self.heartbeat_task = asyncio.create_task(self.heartbeat())
            self.message_handler_task = asyncio.create_task(self.message_handler())
        except Exception as error:
            logger.exception("Error connecting to the gateway")
            raise

    async def close_connection(self):
        if self.ws and not self.ws.closed:
            await self.ws.close()
            logger.info("WebSocket connection closed")
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.message_handler_task:
            self.message_handler_task.cancel()

    async def heartbeat(self):
        while self.ws and not self.ws.closed:
            try:
                logger.debug("Sending heartbeat")
                await self.ws.ping()

                # Log pending requests
                async with self._lock:
                    pending_requests = list(self._pending_requests.keys())
                logger.debug(f"Current pending requests: {pending_requests}")

                await asyncio.sleep(30)
            except Exception:
                logger.exception("Error in heartbeat")
                break

    async def message_handler(self):
        while True:
            try:
                message = await self.ws.recv()
                asyncio.create_task(self._process_message(message))
            except (ConnectionClosedError, ConnectionClosedOK):
                logger.warning("Connection to the gateway was closed.")
                await self.close_connection()
                break
            except Exception:
                logger.exception(f"Error receiving message")
                await self.close_connection()
                break

    async def _process_message(self, message: str):
        async with self._semaphore:
            try:
                data = json.loads(message)
                message_type = data.get("type")
                correlation_id = data.get("correlation_id")
                logger.debug(
                    textwrap.dedent(
                        f"""
                        Received message.
                        Type: {message_type}
                        Correlation ID: {correlation_id}
                    """
                    ).strip()
                )

                if correlation_id and self._pending_requests.get(correlation_id):
                    # Associate the response with the pending request
                    await self._handle_correlated_response(correlation_id, data)
                elif message_type in self.message_handlers:
                    await self.message_handlers[message_type](data)

                    if message_type in self.relevant_message_types():
                        # Trigger the timeout reset event for all pending requests
                        for timeout_event in self._timeout_reset_events.values():
                            timeout_event.set()
                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except Exception:
                logger.exception("Error processing message")

    def relevant_message_types(self) -> List[str]:
        """
        Define which message types should reset the timeout.
        Add all relevant message types that should reset the timeout here.
        """
        return [
            WsLocalTask.GENERATE.value,
            WsLocalTask.EVALUATE.value,
            # Add other message types as needed
        ]

    async def send_message(self, message: Dict[str, Any]):
        if self.ws and not self.ws.closed:
            logger.debug(
                textwrap.dedent(
                    f"""
                    Sending message.
                    Type: {message.get('type')}
                    Correlation ID: {message.get('correlation_id')}
                """
                ).strip()
            )
            await self.ws.send(json.dumps(message, cls=CustomJSONEncoder))
            logger.debug(
                textwrap.dedent(
                    f"""
                    Message sent.
                    Type: {message.get('type')}
                    Correlation ID: {message.get('correlation_id')}
                """
                ).strip()
            )
        else:
            raise ConnectionError("WebSocket connection is not established or closed")

    async def _handle_correlated_response(
        self, correlation_id: str, data: Dict[str, Any]
    ):
        if not correlation_id:
            logger.error("Received message without correlation_id. Ignoring.")
            return

        logger.debug(
            f"Handling correlated response for correlation_id: {correlation_id}"
        )
        async with self._lock:
            response_queue = self._responses.get(correlation_id)
            if response_queue:
                logger.debug(
                    f"Putting data into response queue for correlation_id: {correlation_id}"
                )
                await response_queue.put(data)
                event = self._pending_requests.get(correlation_id)
                if event:
                    logger.debug(f"Setting event for correlation_id: {correlation_id}")
                    event.set()
            else:
                logger.warning(
                    f"No response queue found for correlation_id: {correlation_id}"
                )
                logger.debug(
                    f"Current pending requests: {list(self._pending_requests.keys())}"
                )
                logger.debug(f"Current response queues: {list(self._responses.keys())}")

    async def _generate_correlation_id(self) -> str:
        return str(uuid4())

    async def _register_request(self, correlation_id: str):
        async with self._lock:
            event = asyncio.Event()
            self._pending_requests[correlation_id] = event
            self._timeout_reset_events[correlation_id] = asyncio.Event()
            self._responses[correlation_id] = (
                asyncio.Queue()
            )  # Ensure a queue is created
            logger.debug(f"Registered request with correlation_id: {correlation_id}")

    async def _unregister_request(self, correlation_id: str):
        async with self._lock:
            self._pending_requests.pop(correlation_id, None)
            self._responses.pop(correlation_id, None)
            self._timeout_reset_events.pop(correlation_id, None)
            logger.debug(f"Unregistered request with correlation_id: {correlation_id}")

    async def wait_for_response(
        self, correlation_id: str, timeout: float = 600.0
    ) -> Dict[str, Any]:
        async with self._lock:
            if correlation_id not in self._pending_requests:
                raise ValueError(
                    f"No pending request with correlation_id {correlation_id}."
                )
            event = self._pending_requests[correlation_id]
            timeout_event = self._timeout_reset_events[correlation_id]

        end_time = time.time() + timeout
        while True:
            remaining = end_time - time.time()
            if remaining <= 0:
                logger.error(
                    f"Timeout waiting for response. Correlation ID: {correlation_id}"
                )
                await self._unregister_request(correlation_id)
                raise asyncio.TimeoutError

            try:
                # Check if the WebSocket is closed
                if self.ws.closed:
                    logger.error("WebSocket connection is closed")
                    await self._unregister_request(correlation_id)
                    raise ConnectionError("WebSocket connection is closed")

                # Create tasks
                wait_tasks = [
                    asyncio.create_task(event.wait()),
                    asyncio.create_task(timeout_event.wait()),
                    asyncio.create_task(
                        self.ws.wait_closed()
                    ),  # New task to detect WebSocket closure
                ]

                done, pending = await asyncio.wait(
                    wait_tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel any pending tasks
                for task in pending:
                    task.cancel()

                if wait_tasks[0] in done:
                    response = await self._responses[correlation_id].get()
                    await self._unregister_request(correlation_id)
                    return response["data"]

                if wait_tasks[1] in done:
                    # Reset the timeout countdown
                    end_time = time.time() + timeout
                    timeout_event.clear()
                    logger.debug(f"Timeout reset for correlation_id: {correlation_id}")

                if wait_tasks[2] in done:
                    logger.error(
                        "WebSocket connection closed while waiting for response"
                    )
                    await self._unregister_request(correlation_id)
                    raise ConnectionError(
                        "WebSocket connection closed while waiting for response"
                    )

            except Exception:
                logger.exception("Error waiting for response")
                await self._unregister_request(correlation_id)
                raise

    async def request(self, type: WsServerTask, data: Dict[str, Any] = {}):
        if not self.ws or self.ws.closed:
            await self.connect_to_gateway()

        correlation_id = await self._generate_correlation_id()
        await self._register_request(correlation_id)

        message = {
            "correlation_id": correlation_id,
            "type": type.value,
            "data": data,
        }
        try:
            await self.send_message(message)
            logger.debug(f"Sent request to server. Message: {message}")

            response = await self.wait_for_response(correlation_id)
            logger.debug(f"Received response: {response}")
            return response
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response. Message: {message}")
            raise
        except ConnectionError as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            await self.close_connection()
            raise
        except Exception as e:
            logger.exception(f"Error for request to server. Message: {message}")
            await self.close_connection()
            raise
        finally:
            await self._unregister_request(correlation_id)

    def register_handlers(self, handler_class):
        for attr_name in dir(handler_class):
            attr = getattr(handler_class, attr_name)
            if hasattr(attr, "message_type"):
                self.message_handlers[attr.message_type] = attr

        logger.info(f"Registered message handlers: {self.message_handlers}")
