import os

from ner_graph.pipeline import run_pipeline


def main() -> None:
    project_root = os.path.dirname(__file__)
    run_pipeline(project_root)


if __name__ == "__main__":
    main()
