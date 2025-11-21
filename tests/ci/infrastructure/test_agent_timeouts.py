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
    # Mock get_browser_state_summary to return a minimal state
    from browser_use.browser.views import BrowserStateSummary, TabInfo
    from browser_use.dom.views import SerializedDOMState
    mock_browser_session.get_browser_state_summary = AsyncMock(return_value=BrowserStateSummary(
        url='http://example.com',
        title='Example',
        tabs=[TabInfo(target_id='test-0', url='http://example.com', title='Example')],
        screenshot=None,
        dom_state=SerializedDOMState(_root=None, selector_map={}),
    ))

    agent = Agent(
        task="Test task",
        llm=llm,
        browser_session=mock_browser_session,
        llm_timeout=1,
    )

    # The timeout is caught internally and recorded as an error in the history
    history = await agent.run(max_steps=1)

    assert len(history.errors()) > 0, "Expected an error to be recorded in history"
    error = history.errors()[0]
    assert error is not None, "Expected a non-None error"
    assert "timed out after 1 seconds" in error or "TimeoutError" in error


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
    # Create custom action model that includes extract
    from browser_use.tools.service import Tools
    tools = Tools()
    ActionModel = tools.registry.create_action_model()
    AgentOutputWithActions = AgentOutput.type_with_custom_actions(ActionModel)

    extract_action_output = AgentOutputWithActions.model_validate_json("""
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
    from browser_use.llm.views import ChatInvokeCompletion
    main_llm.ainvoke = AsyncMock(return_value=ChatInvokeCompletion(completion=extract_action_output, usage=None))

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
    # Mock get_browser_state_summary to return a minimal state
    from browser_use.browser.views import BrowserStateSummary, TabInfo
    from browser_use.dom.views import SerializedDOMState
    mock_browser_session.get_browser_state_summary = AsyncMock(return_value=BrowserStateSummary(
        url='http://example.com',
        title='Example',
        tabs=[TabInfo(target_id='test-0', url='http://example.com', title='Example')],
        screenshot=None,
        dom_state=SerializedDOMState(_root=None, selector_map={}),
    ))
    async def mock_extract_clean_markdown(*args, **kwargs):
        return ("Some page content", {
            "final_filtered_chars": 17,
            "original_html_chars": 100,
            "initial_markdown_chars": 50,
            "filtered_chars_removed": 33
        })
    mocker.patch(
        'browser_use.dom.markdown_extractor.extract_clean_markdown',
        side_effect=mock_extract_clean_markdown
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
    # The extract tool should fail due to timeout (error message may vary)
    assert "extract" in error.lower()


@pytest.mark.asyncio
async def test_agent_llm_success(mocker: MockerFixture):
    """
    Test that the agent's LLM call succeeds when it completes within the timeout.
    """
    # Create LLM with proper action model
    from browser_use.tools.service import Tools
    tools = Tools()
    ActionModel = tools.registry.create_action_model()
    AgentOutputWithActions = AgentOutput.type_with_custom_actions(ActionModel)

    llm = create_mock_llm()

    # Mock ainvoke to complete quickly (within timeout)
    async def fast_ainvoke(*args, **kwargs):
        await asyncio.sleep(0.1)  # Complete in 0.1 seconds (well within 1 second timeout)
        # Return a proper done action
        done_action = AgentOutputWithActions.model_validate_json("""
        {
            "thinking": "Task completed",
            "evaluation_previous_goal": "Success",
            "memory": "Done",
            "next_goal": "Finish",
            "action": [
                {
                    "done": {
                        "text": "Task completed successfully",
                        "success": true
                    }
                }
            ]
        }
        """)
        from browser_use.llm.views import ChatInvokeCompletion
        return ChatInvokeCompletion(completion=done_action, usage=None)

    llm.ainvoke.side_effect = fast_ainvoke

    mock_browser_session = mocker.MagicMock(spec=BrowserSession)
    mock_browser_session.id = 'mock_session_id'
    mock_browser_session.agent_focus_target_id = None
    mock_browser_session.browser_profile = mocker.MagicMock()
    mock_browser_session.browser_profile.demo_mode = False
    mock_browser_session.browser_profile.downloads_path = None
    mock_browser_session.llm_screenshot_size = None
    mock_browser_session.cdp_url = None
    mock_browser_session.start = AsyncMock()
    # Mock get_browser_state_summary to return a minimal state
    from browser_use.browser.views import BrowserStateSummary, TabInfo
    from browser_use.dom.views import SerializedDOMState
    mock_browser_session.get_browser_state_summary = AsyncMock(return_value=BrowserStateSummary(
        url='http://example.com',
        title='Example',
        tabs=[TabInfo(target_id='test-0', url='http://example.com', title='Example')],
        screenshot=None,
        dom_state=SerializedDOMState(_root=None, selector_map={}),
    ))

    agent = Agent(
        task="Test task",
        llm=llm,
        browser_session=mock_browser_session,
        llm_timeout=1,  # 1 second timeout
        use_judge=False,  # Disable judge for this test
    )

    # The LLM should complete successfully within the timeout
    history = await agent.run(max_steps=1)

    # No real errors should be recorded (filter out None errors)
    real_errors = [e for e in history.errors() if e is not None]
    assert len(real_errors) == 0, f"Expected no errors in history, but got: {real_errors}"
    # Task should be marked as done
    assert history.is_done(), "Expected task to be marked as done"


@pytest.mark.asyncio
async def test_agent_step_success(mocker: MockerFixture):
    """
    Test that the agent's step execution succeeds when it completes within the timeout.
    """
    # Create proper action model
    from browser_use.tools.service import Tools
    tools = Tools()
    ActionModel = tools.registry.create_action_model()
    AgentOutputWithActions = AgentOutput.type_with_custom_actions(ActionModel)

    # Mock the Agent.step method to complete quickly
    original_step_call_count = 0

    async def fast_step(self, *args, **kwargs):
        nonlocal original_step_call_count
        original_step_call_count += 1
        await asyncio.sleep(0.1)  # Complete in 0.1 seconds (well within 1 second timeout)
        # Manually set the state to simulate a successful step
        self.state.n_steps += 1
        # Create a done action result
        from browser_use.agent.views import ActionResult
        self.state.last_result = [ActionResult(
            extracted_content='Task completed',
            is_done=True,
        )]
        # Set last_model_output to indicate done
        self.state.last_model_output = AgentOutputWithActions.model_validate_json("""
        {
            "thinking": "Task completed",
            "evaluation_previous_goal": "Success",
            "memory": "Done",
            "next_goal": "Finish",
            "action": [
                {
                    "done": {
                        "text": "Task completed successfully",
                        "success": true
                    }
                }
            ]
        }
        """)

    mocker.patch('browser_use.agent.service.Agent.step', fast_step)

    llm = create_mock_llm()

    mock_browser_session = mocker.MagicMock(spec=BrowserSession)
    mock_browser_session.id = 'mock_session_id'
    mock_browser_session.agent_focus_target_id = None
    mock_browser_session.browser_profile = mocker.MagicMock()
    mock_browser_session.browser_profile.demo_mode = False
    mock_browser_session.browser_profile.downloads_path = None
    mock_browser_session.llm_screenshot_size = None
    mock_browser_session.cdp_url = None
    mock_browser_session.start = AsyncMock()
    # Mock get_browser_state_summary to return a minimal state
    from browser_use.browser.views import BrowserStateSummary, TabInfo
    from browser_use.dom.views import SerializedDOMState
    mock_browser_session.get_browser_state_summary = AsyncMock(return_value=BrowserStateSummary(
        url='http://example.com',
        title='Example',
        tabs=[TabInfo(target_id='test-0', url='http://example.com', title='Example')],
        screenshot=None,
        dom_state=SerializedDOMState(_root=None, selector_map={}),
    ))

    agent = Agent(
        task="Test task",
        llm=llm,
        browser_session=mock_browser_session,
        step_timeout=1,  # 1 second timeout
        use_judge=False,  # Disable judge for this test
    )

    # The step should complete successfully within the timeout
    history = await agent.run(max_steps=1)

    # No timeout errors should be recorded (filter out None errors)
    timeout_errors = [e for e in history.errors() if e is not None and 'timed out' in e.lower()]
    assert len(timeout_errors) == 0, f"Expected no timeout errors, but got: {timeout_errors}"
    # Step should have been called
    assert original_step_call_count > 0, "Expected step to be called at least once"


@pytest.mark.asyncio
async def test_extract_tool_success(mocker: MockerFixture):
    """
    Test that the extract tool succeeds when the LLM completes within the timeout.
    """
    # 1. Main LLM that instructs to use the 'extract' tool
    from browser_use.tools.service import Tools
    tools = Tools()
    ActionModel = tools.registry.create_action_model()
    AgentOutputWithActions = AgentOutput.type_with_custom_actions(ActionModel)

    extract_action_output = AgentOutputWithActions.model_validate_json("""
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
    from browser_use.llm.views import ChatInvokeCompletion
    main_llm.ainvoke = AsyncMock(return_value=ChatInvokeCompletion(completion=extract_action_output, usage=None))

    # 2. Fast page_extraction_llm that completes within timeout
    page_extraction_llm = create_mock_llm()

    async def fast_ainvoke(*args, **kwargs):
        await asyncio.sleep(0.1)  # Complete in 0.1 seconds (well within 1 second timeout)
        # Return extracted content
        return ChatInvokeCompletion(completion="Extracted: Some page content", usage=None)

    page_extraction_llm.ainvoke.side_effect = fast_ainvoke

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
    # Ensure a valid URL is returned
    type(mock_browser_session).url = mocker.PropertyMock(return_value="http://example.com")
    mock_browser_session.get_current_page_url = AsyncMock(return_value="http://example.com")
    # Mock get_browser_state_summary to return a minimal state
    from browser_use.browser.views import BrowserStateSummary, TabInfo
    from browser_use.dom.views import SerializedDOMState
    mock_browser_session.get_browser_state_summary = AsyncMock(return_value=BrowserStateSummary(
        url='http://example.com',
        title='Example',
        tabs=[TabInfo(target_id='test-0', url='http://example.com', title='Example')],
        screenshot=None,
        dom_state=SerializedDOMState(_root=None, selector_map={}),
    ))

    async def mock_extract_clean_markdown(*args, **kwargs):
        return ("Some page content", {
            "final_filtered_chars": 17,
            "original_html_chars": 100,
            "initial_markdown_chars": 50,
            "filtered_chars_removed": 33
        })

    mocker.patch(
        'browser_use.dom.markdown_extractor.extract_clean_markdown',
        side_effect=mock_extract_clean_markdown
    )

    # 4. Initialize Agent with timeout and the two LLMs
    agent = Agent(
        task="Test extract success",
        llm=main_llm,
        page_extraction_llm=page_extraction_llm,
        browser_session=mock_browser_session,
        llm_timeout=1,  # 1 second timeout
        use_judge=False,  # Disable judge for this test
    )

    # 5. Run the agent and verify success
    history = await agent.run(max_steps=1)

    # The extract action should succeed
    # Check that we have results and no extraction-related errors (filter out None errors)
    extract_errors = [e for e in history.errors() if e is not None and 'extract' in e.lower()]
    assert len(extract_errors) == 0, f"Expected no extract errors, but got: {extract_errors}"
    # Verify that extraction results are present
    assert len(history.history) > 0, "Expected at least one history entry"
