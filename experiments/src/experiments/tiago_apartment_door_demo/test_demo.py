#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the Tiago door passing demo and exit non-zero with a traceback on failure.
    """
    try:
        from experiments.tiago_apartment_door_demo import demo

        demo.main()
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
