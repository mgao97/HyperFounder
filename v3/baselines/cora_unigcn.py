import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import main

if __name__ == "__main__":
    # UniGCN baseline on Cora. Launched on its own GPU by launch_cora_baselines.sh.
    main(sys.argv[1:] + ["--model", "unigcn", "--dataset", "cora"])
