import sys

from src import search as search_cli


def test_media_both_prints_separate_labeled_result_sets(monkeypatch, capsys):
    monkeypatch.setattr(search_cli, "embed_text", lambda query: [1.0, 0.0])
    monkeypatch.setattr(search_cli, "embed_text_for_audio", lambda query: [0.0, 1.0])
    monkeypatch.setattr(search_cli, "_print_results", lambda label, *args: print(f"{label} RESULTS"))
    monkeypatch.setattr(sys, "argv", ["search", "--query", "dog barking", "--media", "both"])
    search_cli.main()
    output = capsys.readouterr().out
    assert "IMAGE RESULTS" in output
    assert "AUDIO RESULTS" in output
    assert output.index("IMAGE RESULTS") < output.index("AUDIO RESULTS")
