"""Synthetic credential-context regressions; never use trace excerpts or real keys."""

import pytest

from atif_scan import Context, Severity, Status, parse_trace
from atif_scan.data import credentials as c
from atif_scan.detectors.side_channel import CredentialExposure
from atif_scan.evidence.cite import mask as cite_mask
from atif_scan.evidence.cite import trace_secrets

LOOKUP_NAMES = [
    "SettingsKey",
    "EnabledKey",
    "SettingKey",
    "analysis_key",
    "IMPLICIT_METHOD_SCOPE_KEY",
    "ATOMIC_PLUGIN_KEY",
    "consistentHashMergeKey",
    "listMapKey",
    "mergeKeyConsistentHash",
    "routing_key",
    "x-dead-letter-routing-key",
    "descriptionKey",
    "optionsKey",
    "CIRCUIT_HANDLE_KEY",
    "add-foreign-key",
    # Unrelated identifiers exercise structural families, not memorized pairs.
    "requestScopeKey",
    "rendererPluginKey",
    "resourceHandleKey",
    "recordMapKey",
    "tenantSettingsKey",
    "queueRoutingKey",
    "recordMergeKey",
    "tableForeignKey",
    "widgetDescriptionKey",
]
OPAQUE = "OpaqueSynthetic8274"


def trace(text):
    return parse_trace({"steps": [{"source": "agent", "message": text}]})


@pytest.mark.parametrize("name", LOOKUP_NAMES)
@pytest.mark.parametrize("template", ['{name} = "{value}"', '"{name}": "{value}"'])
def test_lookup_names_are_not_exposure_but_still_masked(name, template):
    text = template.format(name=name, value=OPAQUE)
    assert not list(c.find_exposures(text))
    assert list(c.find(text))
    assert OPAQUE in c.values([text])
    assert OPAQUE in trace_secrets(trace(text))
    assert OPAQUE not in c.mask(text)
    assert OPAQUE not in cite_mask(text)
    assert c.mask(OPAQUE, trace_secrets(trace(text))) == "***"
    assert CredentialExposure().evaluate(trace(text), Context()).status == Status.NO_MATCH


@pytest.mark.parametrize(
    "name",
    [
        "SettingApiKey",
        "analysis_secret",
        "routing_auth_key",
        "ATOMIC_PLUGIN_ACCESS_KEY",
        "CIRCUIT_HANDLE_PASSWORD",
        "optionsToken",
        "MERGE_PRIVATE_KEY",
        "unfamiliar_key",
        "ZEPHYR_API_KEY",
        "requestScopeApiKey",
        "rendererPluginSecret",
        "resourceHandleAuthKey",
        "recordMapAccessKey",
        "mapping_key",  # substrings are not structural components
        "scoped_key",
    ],
)
def test_explicit_authentication_and_unknown_names_remain_candidates(name):
    assert list(c.find_exposures(f'{name}="{OPAQUE}"'))


@pytest.mark.parametrize("name", LOOKUP_NAMES)
def test_token_shapes_override_lookup_names(name):
    token = "sk-proj-syntheticCredential8274"
    assert any(f.kind == "token" for f in c.find_exposures(f'"{name}": "{token}"'))


@pytest.mark.parametrize(
    "value",
    [
        "test-api-key",
        "dummy-key",
        "example_api_key",
        "fake-api-key-12345",
        "placeholder-token",
        "sk-test-api-key-123456789",
    ],
)
def test_explicit_dummy_values_not_findings(value):
    assert not list(c.find_exposures(f'api_key="{value}"'))
    assert not list(c.find_exposures(f"--api-key {value}"))


@pytest.mark.parametrize("value", ["test-api-key", "dummy-key", "sk-test-api-key-123456789"])
def test_previously_masked_dummies_stay_masked_everywhere(value):
    text = f'api_key="{value}"'
    assert value in trace_secrets(trace(text))
    assert value not in c.mask(text)
    assert value not in cite_mask(text)
    assert c.mask(value, trace_secrets(trace(text))) == "***"


@pytest.mark.parametrize(
    "value",
    [
        "aB7defg8",
        "testament8274",
        "test_live_real8274",
        "exampleCredential8274",
        "test-api-key-realOpaque8274",
        "dummy-actualOpaque8274",
        "unfamiliarSecret8274",
    ],
)
def test_unknown_and_test_prefixed_secrets_are_detected_and_masked(value):
    text = f'api_key="{value}"'
    assert list(c.find_exposures(text))
    assert list(c.find(text))
    assert value in trace_secrets(trace(text))
    assert c.mask(value, trace_secrets(trace(text))) == "***"
    assert c.mask(text) == 'api_key="***"'


def test_fixture_context_does_not_excuse_arbitrary_credentials():
    text = f'# tests/test_client.py fixture\napi_key="{OPAQUE}"'
    assert list(c.find_exposures(text))
    assert CredentialExposure().evaluate(trace(text), Context()).status == Status.MATCH


@pytest.mark.parametrize("kind", ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY"])
@pytest.mark.parametrize("body", ["...", "…", "<redacted>"])
def test_closed_placeholder_pem_not_finding_but_always_masked(kind, body):
    block = f"-----BEGIN {kind}-----\n{body}\n-----END {kind}-----"
    assert not list(c.find_exposures(block))
    assert list(c.find(block))
    assert block in trace_secrets(trace(block))
    assert c.mask(block) == "[private key]"
    assert cite_mask(block) == "[private key]"


@pytest.mark.parametrize(
    "block",
    [
        "-----BEGIN PRIVATE KEY-----\n...",
        "-----BEGIN PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----\nU3ludGhldGlj\n...\n-----END PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----\nU3ludGhldGlj\n-----END PRIVATE KEY-----",
    ],
)
def test_unclosed_mismatched_or_nonplaceholder_pem_remains_candidate(block):
    assert list(c.find_exposures(block))
    assert c.mask(block) == "[private key]"


@pytest.mark.parametrize(
    "text",
    [
        "TIKTOKEN_CACHE_DIR=/root/cache MAX_TOKENS=5000",
        'SSH_KEY_PATH="/root/.ssh/id" MY_TOKEN=${TOKEN}',
        'api_key="your-key"',
        "token = tok.encode(x)",
        "token_chars = 'ABCDEFGHIJ'",
        "KeyError: 'adapt_engaged_setting'",
        "openssl pkcs12 -password=KEY_PASSWORD -in cert.pem",
    ],
)
def test_existing_code_and_reference_exclusions(text):
    assert not list(c.find_exposures(text))


def test_exposure_detector_version_invalidates_old_results():
    spec = CredentialExposure().spec
    assert spec.version == "4"
    assert spec.severity == Severity.LOW
    assert spec.title == "Possible credential-like value"


@pytest.mark.parametrize(
    "algorithm",
    [
        "sk-ssh-ed25519@openssh.com",
        "sk-ssh-ed25519-cert-v01@openssh.com",
        "sk-ecdsa-sha2-nistp256@openssh.com",
        "sk-ecdsa-sha2-nistp256-cert-v01@openssh.com",
    ],
)
@pytest.mark.parametrize("template", ["{value}", 'api_key="{value}"', "--api-key {value}"])
def test_public_ssh_algorithms_are_not_credentials(algorithm, template):
    text = template.format(value=algorithm)
    assert not list(c.find_exposures(text))
    assert CredentialExposure().evaluate(trace(text), Context()).status == Status.NO_MATCH
    # Existing broad masking stays independent, including partial token-shape matches.
    for found in c.find(text):
        candidate = text[slice(*found.value)]
        assert candidate in trace_secrets(trace(text))
        assert candidate not in cite_mask(text)


@pytest.mark.parametrize(
    "value",
    [
        "sk-ssh-ed25519-cert-v01",
        "sk-ssh-ed25519-cert-v01@untrusted.example",
        "sk-ssh-ed25519-cert-v01@openssh.com.attacker",
        "sk-ecdsa-sha2-nistp256@openssh.com-extra",
        "sk-proj-syntheticCredential8274",
    ],
)
def test_near_ssh_names_and_opaque_tokens_remain_candidates(value):
    assert list(c.find_exposures(value))
    assert c.mask(value) != value


def test_public_algorithm_does_not_hide_adjacent_credential():
    public = "sk-ssh-ed25519-cert-v01@openssh.com"
    token = "sk-proj-syntheticCredential8274"
    text = f"{public}, {token}"
    hits = list(c.find_exposures(text))
    assert len(hits) == 1
    assert text[slice(*hits[0].value)] == token
    assert token not in cite_mask(text)


@pytest.mark.parametrize("value", ["p", "abc", "aB7", "aB7defg", "42", "12345678", "123456789012"])
@pytest.mark.parametrize(
    "template",
    ['api_key="{value}"', '"password": "{value}"', "passwd={value}", "--api-key {value}"],
)
def test_short_and_numeric_values_keep_legacy_boundaries(value, template):
    text = template.format(value=value)
    assert not list(c.find_exposures(text))
    assert not list(c.find(text))
    assert not c.values([text])
    known = trace_secrets(trace(text))
    assert not known
    assert c.mask(text) == text
    # Citation-specific local masks remain broader; no value enters global replacement.
    prompt = f"Keep prompts intact: paragraph abc aB7defg 123456789012. Repeat {value}."
    assert c.mask(prompt, known) == prompt
    assert cite_mask(prompt, known) == prompt


def test_single_character_assignment_does_not_corrupt_prompt():
    recorded = parse_trace(
        {
            "steps": [
                {"source": "user", "message": "Prepare a proper program."},
                {"source": "agent", "message": 'api_key="p"'},
            ]
        }
    )
    known = trace_secrets(recorded)
    assert not known
    assert cite_mask("Prepare a proper program.", known) == "Prepare a proper program."
    assert cite_mask("passwd=abc") == "passwd=abc"


@pytest.mark.parametrize(
    "text",
    [
        'rate_key = "embedding_lr"',
        "token = 'identifier_tail'",
        "GPG_KEY=0123456789ABCDEF0123456789ABCDEF01234567",
        'primary_key = "account_ref"',
    ],
)
def test_main_identifier_and_public_key_exclusions_survive_recovery(text):
    assert not list(c.find_exposures(text))
    assert CredentialExposure().evaluate(trace(text), Context()).status == Status.NO_MATCH


@pytest.mark.parametrize(
    "code",
    [
        "if token==prev_tok:\n    merged += 1",  # TB4 telecom-entity-resolution tokenizer
        "keep = [t for t in toks if token==t]",
        "assert api_key==expected_key",
    ],
)
def test_comparisons_are_not_secret_assignments(code):
    # Regression: `token==prev_tok` read "=prev_tok" as a secret-named value.
    assert not list(c.find_exposures(code))


def test_secret_assignments_still_count_next_to_comparisons():
    assert list(c.find_exposures(f"if a==b:\n    API_TOKEN={OPAQUE}"))
