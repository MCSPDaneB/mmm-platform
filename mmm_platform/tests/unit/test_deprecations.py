"""Tests to catch deprecated Streamlit parameters."""

from pathlib import Path


class TestStreamlitDeprecations:
    """Tests to ensure deprecated Streamlit parameters are not used."""

    def test_no_use_container_width_parameter(self):
        """Ensure use_container_width is not used (deprecated after 2025-12-31).

        Use width='stretch' instead of use_container_width=True.
        Use width='content' instead of use_container_width=False.
        """
        ui_dir = Path(__file__).parent.parent.parent / "ui"

        violations = []
        for py_file in ui_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if "use_container_width" in content:
                # Find line numbers
                for i, line in enumerate(content.splitlines(), 1):
                    if "use_container_width" in line:
                        violations.append(f"{py_file.name}:{i}: {line.strip()}")

        assert not violations, (
            "Found deprecated 'use_container_width' parameter "
            "(use width='stretch' or width='content' instead):\n"
            + "\n".join(violations)
        )
