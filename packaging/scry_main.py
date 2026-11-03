"""PyInstaller entry point for the portable / Store build."""
import multiprocessing

from localsearch.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
