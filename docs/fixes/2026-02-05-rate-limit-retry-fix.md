# Rate Limit Error Handling - Exponential Backoff Implementation

**Date:** 2026-02-05
**Issue:** 500 Internal Server Error due to Gemini API rate limits (429 errors)
**Solution:** Implemented exponential backoff retry logic across all LLM calls

## Problem

The application was experiencing 500 errors when making requests to the `/api/collections/{collection_id}/search/agent` endpoint. The root causes were:

1. **Router Agent**: Rate limit errors when trying to classify queries
2. **Search Operations**: Rate limit errors during local/global/tog/drift search execution
3. **No Retry Logic**: Direct litellm/OpenAI calls in backend services had no retry handling

## Solution Overview

Implemented a **Conservative 3-retry exponential backoff strategy** with delays of 1s, 2s, 4s:

### Changes Made

#### 1. GraphRAG Configuration Updates
**Files:**
- `backend/settings.yaml`
- `graphrag/settings.yaml`

**Changes:**
- Reduced `max_retries` from 10 to 3 (more conservative)
- Added `max_retry_wait: 10.0` to cap maximum retry delay
- Applied to both `default_chat_model` and `default_embedding_model`

```yaml
models:
  default_chat_model:
    retry_strategy: exponential_backoff
    max_retries: 3
    max_retry_wait: 10.0
```

#### 2. Router Agent Retry Logic
**File:** `backend/app/services/router_agent.py`

**Changes:**
- Added `asyncio` and `RateLimitError` imports
- Implemented exponential backoff in `_call_llm()` method
- 3 retries with 1s, 2s, 4s delays
- Logs warning on each retry, error on final failure

**Code:**
```python
async def _call_llm(self, prompt: str):
    """Call LLM API using litellm with exponential backoff on rate limits."""
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            return await acompletion(...)
        except RateLimitError as e:
            if attempt == max_retries:
                logger.error(f"Rate limit exceeded after {max_retries} retries: {e}")
                raise

            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Rate limit hit on router agent (attempt {attempt + 1}/{max_retries + 1}). "
                f"Retrying in {delay}s..."
            )
            await asyncio.sleep(delay)
```

#### 3. Web Search Service Retry Logic
**File:** `backend/app/services/web_search.py`

**Changes:**
- Added `asyncio` and OpenAI's `RateLimitError` imports
- Implemented retry logic in `_call_llm()` method
- Implemented retry logic in `search_streaming()` method
- Same 3-retry strategy with exponential backoff

## Retry Strategy Details

### Configuration
- **Max Retries:** 3 attempts (4 total tries including initial)
- **Initial Delay:** 1 second
- **Backoff Multiplier:** 2x (exponential)
- **Delays:** 1s → 2s → 4s
- **Max Total Delay:** ~7 seconds maximum wait time

### Coverage
The retry logic now covers:

1. ✅ **Router Agent** - Query classification LLM calls
2. ✅ **GraphRAG Search Operations** - Local, Global, ToG, DRIFT searches
3. ✅ **Web Search Synthesis** - Both streaming and non-streaming
4. ✅ **Embedding Operations** - Through GraphRAG config

### Error Handling Flow

```
Request → LLM Call
          ↓ (Rate Limit?)
          ├─ No → Return Response
          └─ Yes → Retry Logic
                   ├─ Attempt 1: Wait 1s, retry
                   ├─ Attempt 2: Wait 2s, retry
                   ├─ Attempt 3: Wait 4s, retry
                   └─ Failed → Return 500 with clear error message
```

## Testing

To verify the fix:

1. **Monitor Logs:**
   ```bash
   # Look for retry warnings in backend logs
   grep "Rate limit hit" logs/*.log
   ```

2. **Expected Behavior:**
   - First rate limit: Wait 1s, retry automatically
   - Second rate limit: Wait 2s, retry automatically
   - Third rate limit: Wait 4s, retry automatically
   - Fourth rate limit: Return clear error to user

3. **Success Indicators:**
   - Warning logs showing retry attempts
   - Successful responses after retries
   - No more raw 500 errors for transient rate limits

## Benefits

1. **Resilience:** Automatically handles transient rate limit errors
2. **User Experience:** Most requests succeed without user intervention
3. **Visibility:** Clear logging of retry attempts for debugging
4. **Conservative:** 3 retries balance success rate vs. response time
5. **Consistent:** Same strategy across all LLM call points

## Future Improvements

Consider if issues persist:

1. **Request Rate Limiting:** Implement request throttling at application level
2. **Quota Management:** Track and display API quota usage
3. **Fallback Strategy:** Auto-fallback to web search when GraphRAG quota exhausted
4. **Queue System:** Queue requests during high load periods
5. **Model Selection:** Use different models with separate quotas

## Related Files

- `graphrag/language_model/providers/litellm/services/retry/exponential_retry.py` - Existing retry implementation for GraphRAG
- `backend/app/routers/search.py` - Search endpoints (now protected by retry logic)
- `backend/app/config.py` - Backend configuration settings
