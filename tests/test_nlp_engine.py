from src.nlp_engine import detect_themes


def test_theme_detection_finds_multiple_themes():
    text = "My manager is supportive, but the workload and deadlines are stressful."
    themes = detect_themes(text)

    assert "Management" in themes
    assert "Workload" in themes
    assert "Stress" in themes
