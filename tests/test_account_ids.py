"""OpenAI/ChatGPT account identifiers. Every id here is synthetic (all-zero UUIDs,
`Synthetic…` org/user/project ids); nothing is a real account."""

from __future__ import annotations

import json
import time

import pytest

from atif_scan import Context, Engine, Status, builtin_detectors, parse_trace, report
from atif_scan.data.account_ids import find, values
from atif_scan.data.model import Channel, Locator
from atif_scan.detectors.account_ids import account_id_detectors
from atif_scan.evidence.cite import cite, mask, trace_secrets

ACCOUNT = "00000000-0000-4000-8000-000000000000"
OTHER = "00000000-0000-4000-8000-0000000000aa"
USER = "user-Synthetic0User0000000001"
ORG = "org-Synthetic0Org00000000001"
PROJ = "proj_Synthetic0Proj0000000001"
NAMESPACE = "https://api.openai.com/auth"

EXPOSED = "observation.account_ids_exposed"
SHAPES = "observation.account_id_shapes"
CLAIMS = "observation.account_claims"


def kinds(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in find(text):
        out.setdefault(f.kind, []).append(text[f.value[0] : f.value[1]])
    return out


def claims(**extra: object) -> dict[str, object]:
    auth = {
        "chatgpt_account_id": ACCOUNT,
        "chatgpt_plan_type": "plus",
        "chatgpt_user_id": USER,
        "user_id": USER,
        "organizations": [{"id": ORG, "is_default": True, "role": "owner"}],
        **extra,
    }
    return {"aud": ["https://example.invalid"], NAMESPACE: auth, "exp": 0}


@pytest.mark.parametrize(
    "text",
    [
        f'{{"chatgpt_account_id": "{ACCOUNT}"}}',
        f"chatgpt-account-id: {ACCOUNT}",
        f"ChatGPT-Account-Id: {ACCOUNT}\r\nUser-Agent: synthetic",
        f"CHATGPT-ACCOUNT-ID:{ACCOUNT}",
        f"https://example.invalid/backend?chatgpt_account_id={ACCOUNT}&x=1",
        f"export CHATGPT_ACCOUNT_ID={ACCOUNT}",
        f"curl -H 'x-chatgpt-account-id: {ACCOUNT}' https://example.invalid",
        f'headers["ChatGPT-Account-Id"] = "{ACCOUNT}"',
        f'("chatgpt-account-id", "{ACCOUNT}")',
        f'{{\\"chatgpt_account_id\\":\\"{ACCOUNT}\\"}}',
        f"{{'chatgptAccountId': '{ACCOUNT}'}}",
        f"account: chatgpt_account_id={ACCOUNT.upper()}",
    ],
)
def test_account_key_with_uuid_is_bound(text):
    found = kinds(text)
    assert [v.lower() for v in found["bound"]] == [ACCOUNT]
    assert "shape" not in found


@pytest.mark.parametrize(
    ("text", "value"),
    [
        (f"OpenAI-Organization: {ORG}", ORG),
        (f"openai-project: {PROJ}", PROJ),
        (f'{{"openai-organization": "{ORG}"}}', ORG),
        (f"OPENAI_ORG_ID={ORG}", ORG),
        (f"OPENAI_PROJECT_ID={PROJ}", PROJ),
        # Response headers can carry an organization slug rather than an `org-` id.
        ("openai-organization: user-abc12345", "user-abc12345"),
        (f'"chatgpt_user_id": "{USER}"', USER),
    ],
)
def test_org_and_project_headers_are_bound(text, value):
    assert kinds(text) == {"bound": [value]}


@pytest.mark.parametrize("render", [json.dumps, str, lambda c: json.dumps(json.dumps(c))])
def test_auth_claim_ids_are_bound_and_plan_type_is_context(render):
    text = render(claims(organization_id=ORG))
    found = kinds(text)
    assert set(found["bound"]) == {ACCOUNT, USER, ORG}
    # chatgpt_user_id and user_id hold the same value at different places: both found.
    assert sorted(found["bound"]).count(USER) == 2
    assert found["context"] == ["plus"]
    assert "shape" not in found


def test_claim_object_without_ids_is_context():
    text = json.dumps({NAMESPACE: {"chatgpt_plan_type": "pro"}})
    found = kinds(text)
    assert found == {"context": [NAMESPACE, "pro"]}


def test_redacted_claim_ids_leave_only_context():
    text = json.dumps(
        claims(
            chatgpt_account_id="<redacted>",
            chatgpt_user_id="[REDACTED]",
            user_id="***",
            organizations=[{"id": "REDACTED_ORG_0000"}],
        )
    )
    assert kinds(text) == {"context": [NAMESPACE, "plus"]}


def test_shape_only_ids_are_low_priority_shapes():
    text = f"Created {PROJ} under {ORG} for {USER}."
    assert kinds(text) == {"shape": [PROJ, ORG, USER]}


@pytest.mark.parametrize(
    "text",
    [
        # Bare UUIDs and non-account ids.
        f"request {ACCOUNT} finished",
        f"x-request-id: {ACCOUNT}",
        f'{{"id": "{ACCOUNT}", "session_id": "{OTHER}", "tool_call_id": "call_{OTHER}"}}',
        f'{{"account_id": "{ACCOUNT}", "user_id": "u12345678", "organization_id": "{OTHER}"}}',
        "x-request-id: req_0123456789abcdefABCDEF0123",
        "commit 0123456789abcdef0123456789abcdef01234567 (HEAD -> main)",
        # Redacted values.
        '{"chatgpt_account_id": "<redacted>"}',
        '{"chatgpt_account_id": "[REDACTED]"}',
        '{"chatgpt_account_id": "***"}',
        "ChatGPT-Account-Id: [REDACTED]",
        "chatgpt-account-id: <redacted>",
        "OpenAI-Organization: ***",
        "openai-project: <redacted>",
        # Prose and hyphenated words.
        "A user-friendly, user-facing, org-wide change for the user-experience team.",
        "user-interfacecomponentslibrary and organization-wide-settings",
        "(org-babel-execute-src-block) in org-mode; cyborg-AAAAAAAAAAAAAAAAAAAAAA1",
        "user-0123 org-42 proj_abc",
        "sk-proj-is-not-a-project-id and project_Synthetic0Proj0000000001",
        # Code, not values.
        'openai_organization=org_id; headers["OpenAI-Organization"] = org_id',
        "chatgpt_account_id = get_account_id(token)",
        "CHATGPT_ACCOUNT_ID=${CHATGPT_ACCOUNT_ID}",
        f"chatgpt_account_ids: [{ACCOUNT}]",
        # The namespace outside a claim object (a URL in docs).
        f"see {NAMESPACE} for the claim namespace",
    ],
)
def test_not_account_ids(text):
    assert not list(find(text))


def test_long_candidates_are_linear():
    start = time.perf_counter()
    for text in ("a_" * 50_000, "chatgpt_" * 20_000, "org-" * 30_000, "{" * 100_000):
        assert not list(find(text))
    unclosed = list(find(f'"{NAMESPACE}": {{' + "{" * 100_000))
    assert [f.kind for f in unclosed] == ["context"]
    assert time.perf_counter() - start < 5  # generous: quadratic scanning takes minutes


def test_values_skip_context():
    text = json.dumps(claims())
    assert values([text]) == {ACCOUNT, USER, ORG}


# Citation masking: the cited excerpt must never show the identifier.


def test_mask_removes_bound_and_shape_ids():
    text = f"ChatGPT-Account-Id: {ACCOUNT}\nOpenAI-Organization: {ORG}\nsaw {PROJ}"
    masked = mask(text)
    for value in (ACCOUNT, ORG, PROJ):
        assert value not in masked
    assert "ChatGPT-Account-Id: ***" in masked


def test_mask_removes_claim_ids_and_known_bare_repeats():
    text = json.dumps(claims())
    masked = mask(text)
    for value in (ACCOUNT, USER, ORG):
        assert value not in masked
    assert ACCOUNT in mask(f"bare {ACCOUNT}")  # unbound on its own
    assert ACCOUNT not in mask(f"bare {ACCOUNT}", frozenset({ACCOUNT}))


def agent(command: str = "", output: str | None = None, message: str = "") -> dict[str, object]:
    step: dict[str, object] = {
        "source": "agent",
        "message": message,
        "tool_calls": [
            {"tool_call_id": "c", "function_name": "bash", "arguments": {"command": command}}
        ],
    }
    if output is not None:
        step["observation"] = {"results": [{"source_call_id": "c", "content": output}]}
    return step


def trace(*steps: dict[str, object]):
    return parse_trace({"schema_version": "ATIF-v1.7", "steps": list(steps)})


def results(t):
    return {d.spec.id: d.evaluate(t, Context()) for d in account_id_detectors()}


def test_registered_as_builtins_with_priorities():
    specs = {d.spec.id: d.spec for d in builtin_detectors()}
    assert specs[EXPOSED].severity > specs[SHAPES].severity
    assert specs[SHAPES].severity == specs[CLAIMS].severity


def test_header_in_tool_output_is_exposed():
    t = trace(agent("cat request.log", f"ChatGPT-Account-Id: {ACCOUNT}\n"))
    r = results(t)
    assert r[EXPOSED].status == Status.MATCH
    (at,) = r[EXPOSED].evidence
    assert at == Locator(0, Channel.OBSERVATION, observation=0, span=(0, 18))
    assert r[SHAPES].status == Status.NO_MATCH
    assert r[CLAIMS].status == Status.NO_MATCH


def test_prompt_and_reasoning_are_searched_too():
    t = trace(
        {"source": "system", "message": f"debug claims: {json.dumps(claims())}"},
        agent("true", "", message=f"using project {PROJ}"),
    )
    r = results(t)
    assert all(r[c].status == Status.MATCH for c in (EXPOSED, SHAPES, CLAIMS))
    assert {at.step for at in r[EXPOSED].evidence} == {0}
    assert {at.step for at in r[SHAPES].evidence} == {1}


def test_clean_missing_and_redacted():
    assert all(d.status == Status.NO_MATCH for d in results(trace(agent("ls", "a.txt"))).values())
    assert all(d.status == Status.UNKNOWN for d in results(trace(agent("ls"))).values())
    redacted = trace(agent("cat h", "chatgpt-account-id: <redacted>\nx-request-id: " + ACCOUNT))
    assert all(d.status == Status.NO_MATCH for d in results(redacted).values())
    assert results(trace())[EXPOSED].status == Status.UNKNOWN


def test_report_and_citation_never_carry_the_value():
    t = trace(agent(f"echo {ACCOUNT}", f'{{"chatgpt_account_id": "{ACCOUNT}"}}'))
    assessments = Engine(builtin_detectors()).evaluate(t)
    output = json.dumps(report(assessments))
    assert ACCOUNT not in output
    assert EXPOSED in output
    (at,) = next(a for a in assessments if a.spec.id == EXPOSED).result.evidence
    assert ACCOUNT in trace_secrets(t)
    # The bare repeat in the command is masked through the trace's known values.
    cited = json.dumps(cite(t, at))
    assert ACCOUNT not in cited
    assert "chatgpt_account_id" in cited
    command = json.dumps(cite(t, Locator(0, Channel.COMMAND, call=0, field=0, span=(0, 4))))
    assert ACCOUNT not in command
