import os
import tempfile
import unittest
from unittest.mock import patch

import secrets_config


class TestSecretsConfig(unittest.TestCase):

    def test_get_db_password(self):
        with patch.dict(os.environ, {"OLMOCR_PG_PASS": "test_pass"}):
            self.assertEqual(secrets_config.get_db_password(), "test_pass")

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(secrets_config.get_db_password(), secrets_config.DEFAULT_DB_PASSWORD)

    def test_get_minio_access_key(self):
        with patch.dict(os.environ, {"OLMOCR_MINIO_ACCESS_KEY": "test_minio_access"}):
            self.assertEqual(secrets_config.get_minio_access_key(), "test_minio_access")

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(secrets_config.get_minio_access_key(), secrets_config.DEFAULT_MINIO_ACCESS_KEY)

    def test_get_minio_secret_key(self):
        with patch.dict(os.environ, {"OLMOCR_MINIO_SECRET_KEY": "test_minio_secret"}):
            self.assertEqual(secrets_config.get_minio_secret_key(), "test_minio_secret")

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(secrets_config.get_minio_secret_key(), secrets_config.DEFAULT_MINIO_SECRET_KEY)

    def test_credentials_are_default(self):
        with patch.dict(
            os.environ,
            {
                "OLMOCR_PG_PASS": secrets_config.UNSAFE_DEFAULT_DB_PASSWORD,
                "OLMOCR_MINIO_ACCESS_KEY": "valid_key",
                "OLMOCR_MINIO_SECRET_KEY": "valid_secret",
            },
        ):
            self.assertTrue(secrets_config.credentials_are_default())

        with patch.dict(
            os.environ,
            {
                "OLMOCR_PG_PASS": "custom_pass",
                "OLMOCR_MINIO_ACCESS_KEY": secrets_config.DEFAULT_MINIO_ACCESS_KEY,
                "OLMOCR_MINIO_SECRET_KEY": "custom_secret",
            },
        ):
            self.assertTrue(secrets_config.credentials_are_default())

        with patch.dict(
            os.environ,
            {
                "OLMOCR_PG_PASS": "custom_pass",
                "OLMOCR_MINIO_ACCESS_KEY": "custom_key",
                "OLMOCR_MINIO_SECRET_KEY": "custom_secret",
            },
        ):
            self.assertFalse(secrets_config.credentials_are_default())

    def test_ensure_dotenv_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = os.path.join(tmpdir, ".env")
            with open(dotenv_path, "w", encoding="utf-8") as f:
                f.write("# Comment line\n")
                f.write("INVALID_LINE_NO_EQUALS\n")
                f.write("TEST_ENV_SECRET_KEY=loaded_secret_value\n")

            with patch("os.path.dirname", return_value=tmpdir):
                with patch.dict(os.environ, {}, clear=True):
                    secrets_config._ensure_dotenv_loaded()
                    self.assertEqual(os.environ.get("TEST_ENV_SECRET_KEY"), "loaded_secret_value")

    def test_ensure_dotenv_loaded_exception_handled(self):
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("Read error")):
                with patch.dict(os.environ, {}, clear=True):
                    # Should not raise exception
                    secrets_config._ensure_dotenv_loaded()


if __name__ == "__main__":
    unittest.main()
