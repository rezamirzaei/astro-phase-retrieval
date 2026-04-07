"""Allow ``python -m src`` to launch the CLI for source-checkout compatibility."""

from src.cli import main

if __name__ == "__main__":
    main()
