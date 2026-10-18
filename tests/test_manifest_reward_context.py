"""Manifest reward evidence reaches CLI context and rules; no nearby reward inference."""

import json

import pytest

from atif_scan.cli import main


@pytest.mark.parametrize(
    ("reward_fields", "reward", "status"),
    [
        ({}, None, "unknown"),
        ({"reward": None}, None, "unknown"),
        ({"reward": 0}, 0, "no_match"),
        ({"reward": 1}, 1, "match"),
    ],
    ids=["absent", "explicit-null", "explicit-zero", "explicit-positive"],
)
def test_manifest_reward_context_and_rule(tmp_path, capsys, reward_fields, reward, status):
    trial = tmp_path / "synthetic-trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    trajectory = agent / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [{"source": "agent", "message": "Finished synthetic work."}],
            }
        )
    )
    verifier = trial / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"reward": 1}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "id": "synthetic-trace",
                        "path": str(trajectory),
                        "task": "configure-git-webserver",
                        "partial": False,
                        **reward_fields,
                    }
                ]
            }
        )
    )
    assert main(["--manifest", str(manifest), "--no-cache", "--format", "json"]) == 0
    item = json.loads(capsys.readouterr().out)["inputs"][0]
    assert item["reward"] == reward
    assert item["input_id"] == "synthetic-trace"
    assert item["task"] == "configure-git-webserver"
    assessments = {row["id"]: row for row in item["assessments"]}
    assert assessments["context.rewarded"]["status"] == status
    rule = assessments["tb21.configure_git_webserver.rewarded_without_ssh"]
    assert rule["status"] == status
    assert not rule["expected_by"]
    # This positive companion is discoverable for direct inputs, not implicit in manifests.
    assert main([str(trajectory), "--no-cache", "--format", "json"]) == 0
    direct = json.loads(capsys.readouterr().out)["inputs"][0]
    assert direct["reward"] == 1
