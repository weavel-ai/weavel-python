from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Any, Union
from uuid import uuid4

from dotenv import load_dotenv
from weavel._worker import Worker
from weavel.types.instances import Session, Span, Trace
from weavel.types.types import DatasetItems

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

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."
        self._worker = Worker(self.api_key, base_url=base_url)

        self.testing = False

    def session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Session:
        """Create a new session for the specified user.

        Args:
            user_id (str): The user ID for the session.
            session_id (str, optional): The session ID. If not provided, a new session ID will be generated.
            created_at (datetime, optional): The created_at for the session. If not provided, the current time will be used.
            metadata (Dict[str, str], optional): Additional metadata for the session.

        Returns:
            Session: The session object.

        """
        if user_id is None and session_id is None:
            raise ValueError("user_id or session_id must be provided.")

        if session_id is None:
            session_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        session = Session(
            user_id=user_id,
            session_id=session_id,
            created_at=created_at,
            metadata=metadata,
            weavel_client=self._worker,
        )
        if self.testing:
            return session

        if user_id is not None:
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
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ref_record_id: Optional[str] = None,
    ) -> Trace:
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

            if isinstance(inputs, str):
                inputs = {"_RAW_VALUE_": inputs}
            if isinstance(outputs, str):
                outputs = {"_RAW_VALUE_": outputs}

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
            return Trace(
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
            return Trace(record_id=record_id, weavel_client=self._worker)

    def span(
        self,
        record_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        name: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
    ):
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

            if isinstance(inputs, str):
                inputs = {"_RAW_VALUE_": inputs}
            if isinstance(outputs, str):
                outputs = {"_RAW_VALUE_": outputs}

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
            return Span(
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
            return Span(observation_id=observation_id, weavel_client=self._worker)

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
        dataset_name: str,
        description: Optional[str] = None,
    ) -> None:
        """Upload a dataset to the Weavel service.

        Args:
            dataset_name (str): The name of the dataset.
            description (str): The description of the dataset.
        """
        if self.testing:
            return

        self._worker.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )

    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Fetch the dataset from the Weavel service.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            Dict[str, Any]: The dataset and dataset items.
        """
        if self.testing:
            return {}

        return self._worker.fetch_dataset(dataset_name)

    def create_dataset_items(
        self,
        dataset_name: str,
        items: Union[List[Dict[str, Any]], List[DatasetItems]],
    ) -> None:
        """Upload dataset items to the Weavel service.

        Args:
            dataset_name (str): The name of the dataset.
            items (List[Dict[str, Any]]): The dataset items to upload. keys: inputs (required, Union[str, Dict[str, str]]), outputs (optional, Union[str, Dict[str, str]]), metadata (optional, Dict[str, str]).
        """
        if self.testing:
            return

        for item in items:
            if not isinstance(item, DatasetItems):
                item = DatasetItems(**item)

        self._worker.create_dataset_items(dataset_name, items)

    def test_result(
        self,
        created_at: datetime,
        name: str,
        test_uuid: str,
        dataset_item_uuid: str,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        if isinstance(inputs, str):
            inputs = {"_RAW_VALUE_": inputs}
        if isinstance(outputs, str):
            outputs = {"_RAW_VALUE_": outputs}
        if self.testing:
            self._worker.capture_test_observation(
                created_at=created_at,
                name=name,
                test_uuid=test_uuid,
                dataset_item_uuid=dataset_item_uuid,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
        return

    def test(
        self,
        func: callable,
        dataset_name: str,
        tags: Optional[List[str]] = None,
    ):
        """Test the function with the dataset.
        It must be used in the jupyter notebook environment.

        Args:
            func (callable): The function to test.
            dataset_name (str): The name of the dataset.
            tags: (List[str], optional): The tags for the test.
            mock_inputs (Dict[str, Any], optional): The mock input variables to run the function.
        """
        import nest_asyncio

        nest_asyncio.apply()

        # fetch dataset from server
        datasets: List[Dict[str, Any]] = self._worker.get_test_dataset(dataset_name)

        if not datasets:
            raise ValueError(f"Dataset {dataset_name} not found.")

        # create test instance in database
        test_uuid = str(uuid4())
        self._worker.create_test(test_uuid, dataset_name, tags)

        self.testing = True
        self._worker.testing = True

        def _runner(
            func: callable, dataset_item: Dict, dataset_item_uuid: str, test_uuid: str
        ):
            # run the function and capture the result
            if "_RAW_VALUE_" in dataset_item:
                result = func(dataset_item["_RAW_VALUE_"])
            else:
                result = func(**dataset_item)
            # if result is not Dict, convert it to Dict
            if isinstance(result, str):
                result = {"_RAW_VALUE_": result}
            elif not isinstance(result, dict):
                result = {"_RAW_VALUE_": str(result)}

            self.test_result(
                created_at=datetime.now(timezone.utc),
                name=dataset_name,
                test_uuid=test_uuid,
                dataset_item_uuid=dataset_item_uuid,
                inputs=dataset_item,
                outputs=result,
            )

        async def _arunner(
            func: callable, dataset_item: Dict, dataset_item_uuid: str, test_uuid: str
        ):
            if "_RAW_VALUE_" in dataset_item:
                result = await func(dataset_item["_RAW_VALUE_"])
            else:
                result = await func(**dataset_item)

            if isinstance(result, str):
                result = {"_RAW_VALUE_": result}
            elif not isinstance(result, dict):
                result = {"_RAW_VALUE_": str(result)}

            self.test_result(
                created_at=datetime.now(timezone.utc),
                name=dataset_name,
                test_uuid=test_uuid,
                dataset_item_uuid=dataset_item_uuid,
                inputs=dataset_item,
                outputs=result,
            )

        async def run_async(func, dataset):
            coros = [
                _arunner(func, data["inputs"], data["uuid"], test_uuid)
                for data in dataset
            ]
            return await asyncio.gather(*coros)

        def run_threaded(func, dataset):
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _runner, func, data["inputs"], data["uuid"], test_uuid
                    )
                    for data in dataset
                ]
                return [future.result() for future in futures]

        print("Testing...")
        if asyncio.iscoroutinefunction(func):
            asyncio.run(run_async(func, datasets))
        else:
            run_threaded(func, datasets)

        print("Test finished. Start Logging...")

        self._worker.testing = False
        self.testing = False

        self._worker.flush()
        print("Test completed.")

    def close(self):
        """Close the client connection."""
        self._worker.stop()

    def flush(self):
        """Flush the buffer."""
        self._worker.flush()
