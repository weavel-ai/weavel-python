from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from datetime import datetime, timezone
import time
from typing import Callable, Dict, List, Optional, Any, Union
from uuid import uuid4

from dotenv import load_dotenv
from weavel._worker import Worker

# from weavel.types.instances import Session, Span, Trace
from weavel.object_clients import SessionClient, SpanClient, TraceClient
from weavel.types.datasets import Dataset, DatasetItem

load_dotenv()


class Weavel:
    """Client for interacting with the Weavel service.

    This class provides methods for creating and managing traces, tracking user actions,
    and closing the client connection.

    Args:
        api_key (str, optional): The API key for authenticating with the Weavel service.
            If not provided, the API key will be retrieved from the environment variable
            WEAVEL_API_KEY.

    Attributes:
        api_key (str): The API key used for authentication.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retry: Optional[int] = 3,
        flush_interval: Optional[int] = 60,
        flush_batch_size: Optional[int] = 20,
    ):
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."
        self._worker = Worker(
            self.api_key,
            base_url=base_url,
            max_retry=max_retry,
            flush_interval=flush_interval,
            flush_batch_size=flush_batch_size,
        )

        self.testing = False

    def session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "SessionClient":
        """Create a new session for the specified user.

        Args:
            user_id (str): The user ID for the session.
            session_id (str, optional): The session ID. If not provided, a new session ID will be generated.
            created_at (datetime, optional): The created_at for the session. If not provided, the current time will be used.
            metadata (Dict[str, str], optional): Additional metadata for the session.

        Returns:
            SessionClient: The session object.

        """
        # if user_id is None and session_id is None:
        #     raise ValueError("user_id or session_id must be provided.")

        if session_id is None:
            session_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        session = SessionClient(
            user_id=user_id,
            session_id=session_id,
            created_at=created_at,
            metadata=metadata,
            weavel_client=self._worker,
        )

        if self.testing:
            return session

        # if user_id is not None:
        self._worker.open_session(
            session_id=session_id,
            created_at=created_at,
            user_id=user_id,
            metadata=metadata,
        )
        return session

    def identify(self, user_id: str, properties: Dict[str, Any]):
        """Identify a user with the specified properties.

        You can save any user information you know about a user for user_id, such as their email, name, or phone number in dict format.
        Properties will be updated for every call.
        """
        if self.testing:
            return
        self._worker.identify_user(user_id, properties)

    def trace(
        self,
        session_id: Optional[str] = None,
        record_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        name: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ref_record_id: Optional[str] = None,
    ) -> TraceClient:
        """Create a new trace record or fetch an existing one

        Args:
            session_id (str, optional): The session ID for the trace.
            record_id (str, optional): The record ID. If not provided, a new record ID will be generated.
            created_at (datetime, optional): The created_at for the trace. If not provided, the current time will be used.
            name (str, optional): The name of the trace.
            inputs (Dict[str, Any], optional): The inputs for the trace.
            outputs (Dict[str, Any], optional): The outputs for the trace.
            metadata (Dict[str, Any], optional): Additional metadata for the trace.
            ref_record_id (str, optional): The record ID to reference.

        """

        if session_id is None and record_id is None:
            raise ValueError("session_id or record_id must be provided.")

        if session_id is not None and name is None:
            raise ValueError(
                "If you want to create a new trace, you must provide a name."
            )

        if session_id is not None:
            # Create a new trace
            if record_id is None:
                record_id = str(uuid4())
            if created_at is None:
                created_at = datetime.now(timezone.utc)

            if not self.testing:
                self._worker.capture_trace(
                    session_id=session_id,
                    record_id=record_id,
                    created_at=created_at,
                    name=name,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                    ref_record_id=ref_record_id,
                )
            return TraceClient(
                session_id=session_id,
                record_id=record_id,
                created_at=created_at,
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                weavel_client=self._worker,
            )
        else:
            # Fetch an existing trace
            return TraceClient(record_id=record_id, weavel_client=self._worker)

    def span(
        self,
        record_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        name: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
    ) -> SpanClient:
        # if record_id, observation_id, parent_observation_id
        # None, None, None -> ValueError
        # None, Value -> Fetch
        # Value -> Create
        # None, -, Value -> Create

        if not record_id and not observation_id and not parent_observation_id:
            raise ValueError(
                "One of the record_id, observation_id, or parent_observation_id must be provided."
            )

        if (
            record_id or (record_id is None and parent_observation_id)
        ) and name is None:
            raise ValueError(
                "If you want to create a new span, you must provide a name."
            )

        if record_id is not None or (
            record_id is None and parent_observation_id is not None
        ):
            # Create a new span
            if observation_id is None:
                observation_id = str(uuid4())
            if created_at is None:
                created_at = datetime.now(timezone.utc)

            if not self.testing:
                self._worker.capture_span(
                    record_id=record_id,
                    observation_id=observation_id,
                    created_at=created_at,
                    name=name,
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                    parent_observation_id=parent_observation_id,
                )
            return SpanClient(
                record_id=record_id,
                observation_id=observation_id,
                created_at=created_at,
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
                weavel_client=self._worker,
            )
        else:
            # Fetch an existing span
            return SpanClient(observation_id=observation_id, weavel_client=self._worker)

    def track(
        self,
        session_id: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
        ref_record_id: Optional[str] = None,
    ) -> None:
        """Track an event for the specified session.

        Args:
            session_id (str): The session ID for the event.
            name (str): The name of the event.
            properties (Dict[str, Any], optional): Additional properties for the event.

        """
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        if not self.testing:
            self._worker.capture_track_event(
                session_id=session_id,
                record_id=record_id,
                created_at=created_at,
                name=name,
                properties=properties,
                metadata=metadata,
                ref_record_id=ref_record_id,
            )

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> None:
        """Upload a dataset to the Weavel service.

        Args:
            name (str): The name of the dataset.
            description (str): The description of the dataset.
        """
        if self.testing:
            return

        self._worker.create_dataset(
            name=name,
            description=description,
        )

    def get_dataset(self, name: str) -> Dataset:
        """
        Retrieves a dataset with the given name.

        Args:
            name (str): The name of the dataset to retrieve.

        Returns:
            Dataset: The retrieved dataset.
        """
        if self.testing:
            return {}

        return self._worker.fetch_dataset(name)

    def create_dataset_items(
        self,
        dataset_name: str,
        items: Union[List[Dict[str, Any]], List[DatasetItem]],
    ) -> None:
        """Upload dataset items to the Weavel service.

        Args:
            dataset_name (str): The name of the dataset.
            items (Union[List[Dict[str, Any]], List[DatasetItem]]): The dataset items to upload.
        """
        if self.testing:
            return

        for item in items:
            if not isinstance(item, DatasetItem):
                item = DatasetItem(**item)

        self._worker.create_dataset_items(dataset_name, items)

    def test(
        self,
        func: Callable,
        dataset_name: str,
        batch_size: int = 50,
        delay: int = 10,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Test the function with the dataset.
        It must be used in the jupyter notebook environment.

        Args:
            func (callable): The function to test.
            dataset_name (str): The name of the dataset.
            tags: (List[str], optional): The tags for the test.
            batch_size (int, optional): The batch size for the test. Default is 50.
            delay (int, optional): The delay (in seconds) between each batch. Default is 10.
            mock_inputs (Dict[str, Any], optional): The mock input variables to run the function.
        """
        import nest_asyncio

        nest_asyncio.apply()

        # fetch dataset from server
        dataset: Dataset = self._worker.fetch_dataset(dataset_name)

        # create test instance in database
        test_uuid = str(uuid4())
        self._worker.create_test(test_uuid, dataset_name, tags)

        self.testing = True
        self._worker.testing = True

        def _runner(
            func: callable,
            inputs: Union[Dict[str, Any], List[Any], str],
            dataset_item_uuid: str,
            test_uuid: str,
        ):
            # run the function and capture the result
            if isinstance(inputs, str):
                result = func(inputs)
            else:
                result = func(**inputs)

            if not isinstance(result, dict) and not isinstance(result, str):
                result = str(result)

            self._worker.capture_test_observation(
                created_at=datetime.now(timezone.utc),
                name=dataset_name,
                test_uuid=test_uuid,
                dataset_item_uuid=dataset_item_uuid,
                inputs=inputs,
                outputs=result,
            )

        async def _arunner(
            func: Callable,
            inputs: Union[Dict[str, Any], List[Any], str],
            dataset_item_uuid: str,
            test_uuid: str,
        ):
            # if "_RAW_VALUE_" in inputs:
            if isinstance(inputs, str):
                result = await func(inputs)
            else:
                result = await func(**inputs)

            if not isinstance(result, dict) and not isinstance(result, str):
                result = str(result)

            self._worker.capture_test_observation(
                created_at=datetime.now(timezone.utc),
                name=dataset_name,
                test_uuid=test_uuid,
                dataset_item_uuid=dataset_item_uuid,
                inputs=inputs,
                outputs=result,
            )

        async def run_async(func: Callable, dataset_items: List[DatasetItem]):
            for i in range(0, len(dataset_items), batch_size):
                batch = list(islice(dataset_items, i, i + batch_size))
                coros = [
                    _arunner(func, data.inputs, data.uuid, test_uuid) for data in batch
                ]
                await asyncio.gather(*coros)
                if i + batch_size < len(dataset_items):
                    await asyncio.sleep(delay)

        def run_threaded(func, dataset_items: List[DatasetItem]):
            with ThreadPoolExecutor() as executor:
                for i in range(0, len(dataset_items), batch_size):
                    batch = list(islice(dataset_items, i, i + batch_size))
                    futures = [
                        executor.submit(
                            _runner, func, data.inputs, data.uuid, test_uuid
                        )
                        for data in batch
                    ]
                    for future in futures:
                        future.result()
                    if i + batch_size < len(dataset_items):
                        time.sleep(delay)

        print("Testing...")
        if asyncio.iscoroutinefunction(func):
            asyncio.run(run_async(func, dataset.items))
        else:
            run_threaded(func, dataset.items)

        print("Test finished. Start Logging...")

        self._worker.testing = False
        self.testing = False

        self._worker.flush()
        print("Test completed.")
        
    def prompt_optimization(
        self,
        dataset_name: str,
        model: str,
        initial_prompt: Optional[str] = None,
    ):
        """
        Prompt optimization is a method to optimize the prompt for the given model.
        It will return the optimized prompt for the model.

        Args:
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
            initial_prompt (str, optional): The initial prompt for the model.
        """
        if self.testing:
            return
        
        self._worker.prompt_optimization(
            dataset_name=dataset_name,
            model=model,
            initial_prompt=initial_prompt,
        )

    def close(self):
        """Close the client connection."""
        self._worker.stop()

    def flush(self):
        """Flush the buffer."""
        self._worker.flush()
