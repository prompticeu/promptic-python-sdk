"""Tests for CLI config management."""

from promptic_sdk.cli.config import load_config, save_ai_application, save_config


class TestConfig:
    def test_load_config_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        monkeypatch.setenv("PROMPTIC_ENDPOINT", "https://test.com")
        config = load_config()
        assert config is not None
        assert config.api_key == "pk_test"
        assert config.endpoint == "https://test.com"

    def test_load_config_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        monkeypatch.delenv("PROMPTIC_ENDPOINT", raising=False)
        monkeypatch.setattr("promptic_sdk.cli.config._read_config_file", dict)
        config = load_config()
        assert config is not None
        assert config.endpoint == "https://promptic.eu"

    def test_load_config_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("PROMPTIC_API_KEY", raising=False)
        monkeypatch.delenv("PROMPTIC_ENDPOINT", raising=False)
        # Ensure no config file interferes
        monkeypatch.setattr("promptic_sdk.cli.config._read_config_file", dict)
        config = load_config()
        assert config is None

    def test_save_and_load_config(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.toml"
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_DIR", tmp_path)
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_FILE", config_file)
        monkeypatch.delenv("PROMPTIC_API_KEY", raising=False)
        monkeypatch.delenv("PROMPTIC_ENDPOINT", raising=False)

        save_config("pk_saved", "https://saved.com")
        assert config_file.exists()

        config = load_config()
        assert config is not None
        assert config.api_key == "pk_saved"
        assert config.endpoint == "https://saved.com"

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.toml"
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_DIR", tmp_path)
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_FILE", config_file)

        save_config("pk_file", "https://file.com")
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_env")

        config = load_config()
        assert config is not None
        assert config.api_key == "pk_env"
        # Endpoint falls back to file since no env var set
        assert config.endpoint == "https://file.com"

    def test_config_file_permissions(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.toml"
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_DIR", tmp_path)
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_FILE", config_file)

        save_config("pk_test", "https://test.com")
        # Check file is owner-only readable
        assert oct(config_file.stat().st_mode & 0o777) == "0o600"

    def test_ai_application_env_takes_precedence_over_legacy_workspace_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        monkeypatch.setenv("PROMPTIC_AI_APPLICATION_ID", "app-current")
        monkeypatch.setenv("PROMPTIC_WORKSPACE_ID", "app-legacy")
        monkeypatch.setattr("promptic_sdk.cli.config._read_config_file", dict)

        config = load_config()

        assert config is not None
        assert config.ai_application_id == "app-current"
        assert config.workspace_id == "app-current"

    def test_loads_legacy_workspace_id_from_config(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        monkeypatch.delenv("PROMPTIC_AI_APPLICATION_ID", raising=False)
        monkeypatch.delenv("PROMPTIC_WORKSPACE_ID", raising=False)
        monkeypatch.setattr(
            "promptic_sdk.cli.config._read_config_file",
            lambda: {"workspace_id": "app-legacy"},
        )

        config = load_config()

        assert config is not None
        assert config.ai_application_id == "app-legacy"

    def test_save_ai_application_replaces_legacy_key(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.toml"
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_DIR", tmp_path)
        monkeypatch.setattr("promptic_sdk.cli.config._CONFIG_FILE", config_file)
        config_file.write_text('workspace_id = "app-legacy"\n')

        save_ai_application("app-current")

        assert config_file.read_text() == 'ai_application_id = "app-current"\n'
