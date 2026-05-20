"""Shared utilities: logging setup and WRDS connection context manager."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator

import psycopg2
import psycopg2.extensions


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with INFO level and a standard format.

    If the named logger already has handlers, it is returned as-is to avoid
    duplicate log output when the function is called multiple times.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextlib.contextmanager
def wrds_connection(
    host: str = "wrds-pgdata.wharton.upenn.edu",
    port: int = 9737,
    dbname: str = "wrds",
    user: str = "coleginter",
) -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager for a psycopg2 WRDS connection using ~/.pgpass.

    The password is read automatically from ~/.pgpass; no explicit password
    argument is needed.

    Args:
        host: WRDS PostgreSQL hostname.
        port: WRDS PostgreSQL port.
        dbname: Database name.
        user: WRDS username.

    Yields:
        An open psycopg2 connection.

    Raises:
        ConnectionError: If the connection cannot be established.
    """
    logger = get_logger(__name__)
    conn: psycopg2.extensions.connection | None = None
    try:
        logger.info("Connecting to WRDS at %s:%s db=%s user=%s", host, port, dbname, user)
        conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user)
        logger.info("WRDS connection established")
        yield conn
    except psycopg2.OperationalError as exc:
        raise ConnectionError(
            f"Could not connect to WRDS ({host}:{port}/{dbname} as {user}). "
            "Ensure ~/.pgpass contains the entry: "
            f"{host}:{port}:{dbname}:{user}:<password>"
        ) from exc
    finally:
        if conn is not None and not conn.closed:
            conn.close()
            logger.info("WRDS connection closed")
