import asyncio
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentOutput
from browser_use.browser.session import BrowserSession
from tests.ci.conftest import create_mock_llm


@pytest.mark.asyncio
async def test_agent_llm_timeout(mocker: MockerFixture):
    """
    Test that the agent's LLM call respects the llm_timeout.
    """
    llm = create_mock_llm()

    # Mock ainvoke to sleep longer than the timeout
    async def slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(2)
        # This part will not be reached if timeout is working
        return AsyncMock()

    llm.ainvoke.side_effect = slow_ainvoke

    mock_browser_session = mocker.MagicMock(spec=BrowserSession)
    mock_browser_session.id = 'mock_session_id'
    mock_browser_session.agent_focus_target_id = None
    mock_browser_session.browser_profile = mocker.MagicMock()
    # Agent.__init__ requires various attributes on browser_session
    mock_browser_session.browser_profile.demo_mode = False
    mock_browser_session.browser_profile.downloads_path = None
    mock_browser_session.llm_screenshot_size = None
    mock_browser_session.cdp_url = None
    mock_browser_session.start = AsyncMock()

    agent = Agent(
        task="Test task",
        llm=llm,
        browser_session=mock_browser_session,
        llm_timeout=1,
    )

    with pytest.raises(asyncio.TimeoutError):
        await agent.run(max_steps=1)


@pytest.mark.asyncio
async def test_agent_step_timeout(mocker: MockerFixture):
    """
    Test that the agent's step execution respects the step_timeout.
    """
    # Mock the Agent.step method to be slow
    async def slow_step(self, *args, **kwargs):
        await asyncio.sleep(2)

    mocker.patch('browser_use.agent.service.Agent.step', slow_step)

    # The mock LLM will return a 'done' action immediately
    llm = create_mock_llm()

    mock_browser_session = mocker.MagicMock(spec=BrowserSession)
    mock_browser_session.id = 'mock_session_id'
    mock_browser_session.agent_focus_target_id = None
    mock_browser_session.browser_profile = mocker.MagicMock()
    # Agent.__init__ requires various attributes on browser_session
    mock_browser_session.browser_profile.demo_mode = False
    mock_browser_session.browser_profile.downloads_path = None
    mock_browser_session.llm_screenshot_size = None
    mock_browser_session.cdp_url = None
    mock_browser_session.start = AsyncMock()


    agent = Agent(
        task="Test task",
        llm=llm,
        browser_session=mock_browser_session,
        step_timeout=1,
    )

    history = await agent.run(max_steps=1)

    # The timeout is caught internally and recorded as an error in the history
    assert len(history.errors()) > 0
    assert history.errors()[0] is not None
    assert "timed out after 1 seconds" in history.errors()[0]


@pytest.mark.asyncio
async def test_extract_tool_timeout(mocker: MockerFixture):
    """
    Test that the extract tool respects the llm_timeout from the Agent.
    """
    # 1. Main LLM that instructs to use the 'extract' tool
    extract_action_output = AgentOutput.model_validate_json("""
    {
        "thinking": "I need to extract data.",
        "evaluation_previous_goal": "I will extract data.",
        "memory": "About to extract data.",
        "next_goal": "Extract data.",
        "action": [
            {
                "extract": {
                    "query": "Extract all the text from the page."
                }
            }
        ]
    }
    """)
    main_llm = create_mock_llm()
    main_llm.ainvoke = AsyncMock(return_value=AsyncMock(completion=extract_action_output))

    # 2. Slow page_extraction_llm that will be used by the 'extract' tool
    page_extraction_llm = create_mock_llm()
    async def slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(2)
    page_extraction_llm.ainvoke.side_effect = slow_ainvoke

    # 3. Mock the browser session to return some content for extraction
    mock_browser_session = mocker.MagicMock(spec=BrowserSession)
    mock_browser_session.id = 'mock_session_id'
    mock_browser_session.agent_focus_target_id = None
    mock_browser_session.browser_profile = mocker.MagicMock()
    mock_browser_session.browser_profile.demo_mode = False
    mock_browser_session.browser_profile.downloads_path = None
    mock_browser_session.llm_screenshot_size = None
    mock_browser_session.cdp_url = None
    mock_browser_session.start = AsyncMock()
    # Ensure a valid URL is returned, and other methods are mocked
    type(mock_browser_session).url = mocker.PropertyMock(return_value="http://example.com")
    mock_browser_session.get_current_page_url = AsyncMock(return_value="http://example.com")
    mocker.patch(
        'browser_use.dom.markdown_extractor.extract_clean_markdown',
        return_value=("Some page content", {"final_filtered_chars": 17})
    )

    # 4. Initialize Agent with a short timeout and the two LLMs
    agent = Agent(
        task="Test extract timeout",
        llm=main_llm,
        page_extraction_llm=page_extraction_llm,
        browser_session=mock_browser_session,
        llm_timeout=1,
    )

    # 5. Run the agent and assert the timeout error
    history = await agent.run(max_steps=1)

    assert len(history.errors()) > 0, "Expected an error to be recorded in history"
    error = history.errors()[0]
    assert error is not None, "Expected a non-None error"
    # The TimeoutError from asyncio.wait_for is caught and re-raised as a RuntimeError
    assert "RuntimeError" in error or "TimeoutError" in error
