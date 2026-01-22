"""
Base agent class with LangSmith tracing integration.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from langsmith import traceable

from src.config.settings import get_settings
from src.services.llm_service import LLMService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents with LangSmith tracing.

    Provides:
    - LLM service integration
    - Automatic tracing with LangSmith
    - Common error handling
    - State management utilities
    """

    def __init__(
        self,
        name: str,
        llm_service: Optional[LLMService] = None,
    ):
        self.name = name
        self.llm = llm_service or LLMService()
        self.settings = get_settings()

        # Configure LangSmith tracing if enabled
        self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Configure LangSmith tracing based on settings."""
        import os

        if self.settings.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.settings.langsmith_project

            langsmith_key = self.settings.langsmith_api_key.get_secret_value()
            if langsmith_key:
                os.environ["LANGCHAIN_API_KEY"] = langsmith_key

            if self.settings.langsmith_endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = self.settings.langsmith_endpoint

    @traceable(name="agent_execution", run_type="chain")
    async def execute(self, state: dict) -> dict:
        """
        Execute agent with full LangSmith tracing.

        Args:
            state: Input state dictionary

        Returns:
            Updated state dictionary
        """
        logger.info(f"Agent '{self.name}' starting execution")

        try:
            # Add metadata for tracing
            state["_agent_name"] = self.name
            state["_agent_started"] = True

            # Execute implementation
            result = await self._execute_impl(state)

            # Mark completion
            result["_agent_completed"] = True
            logger.info(f"Agent '{self.name}' completed successfully")

            return result

        except Exception as e:
            logger.error(f"Agent '{self.name}' failed: {e}")
            state["error"] = str(e)
            state["_agent_failed"] = True
            raise

    @abstractmethod
    async def _execute_impl(self, state: dict) -> dict:
        """
        Implementation of agent logic.

        Override this method in subclasses.

        Args:
            state: Input state dictionary

        Returns:
            Updated state dictionary
        """
        pass

    def validate_state(self, state: dict, required_fields: list[str]) -> bool:
        """
        Validate that required fields are present in state.

        Args:
            state: State dictionary
            required_fields: List of required field names

        Returns:
            True if all fields present

        Raises:
            ValueError: If required fields are missing
        """
        missing = [f for f in required_fields if f not in state]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return True

    def update_state(self, state: dict, updates: dict) -> dict:
        """
        Update state with new values.

        Args:
            state: Current state
            updates: Dictionary of updates

        Returns:
            Updated state
        """
        new_state = state.copy()
        new_state.update(updates)
        return new_state

    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ) -> str:
        """
        Call LLM with tracing.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            provider: LLM provider
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        return await self.llm.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            provider=provider,
            **kwargs,
        )


class AgentState:
    """
    Helper class for managing agent state.

    Provides type-safe access to state fields.
    """

    def __init__(self, state: dict):
        self._state = state

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._state[key] = value

    def has(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self._state

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self._state.copy()

    @property
    def user_input(self) -> str:
        return self._state.get("user_input", "")

    @property
    def input_type(self) -> str:
        return self._state.get("input_type", "text")

    @property
    def extracted_properties(self) -> Optional[dict]:
        return self._state.get("extracted_properties")

    @extracted_properties.setter
    def extracted_properties(self, value: dict) -> None:
        self._state["extracted_properties"] = value

    @property
    def database_matches(self) -> list:
        return self._state.get("database_matches", [])

    @database_matches.setter
    def database_matches(self, value: list) -> None:
        self._state["database_matches"] = value

    @property
    def match_confidence(self) -> float:
        return self._state.get("match_confidence", 0.0)

    @match_confidence.setter
    def match_confidence(self, value: float) -> None:
        self._state["match_confidence"] = value

    @property
    def live_search_results(self) -> list:
        return self._state.get("live_search_results", [])

    @live_search_results.setter
    def live_search_results(self, value: list) -> None:
        self._state["live_search_results"] = value

    @property
    def final_results(self) -> list:
        return self._state.get("final_results", [])

    @final_results.setter
    def final_results(self, value: list) -> None:
        self._state["final_results"] = value

    @property
    def error(self) -> Optional[str]:
        return self._state.get("error")

    @error.setter
    def error(self, value: str) -> None:
        self._state["error"] = value
