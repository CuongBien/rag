import os

from ner_graph.pipeline import run_pipeline


def main() -> None:
    # project_root: directory containing this script (repo root for data/ resolution).
    project_root = os.path.dirname(__file__)
    run_pipeline(project_root)


if __name__ == "__main__":
    main()
