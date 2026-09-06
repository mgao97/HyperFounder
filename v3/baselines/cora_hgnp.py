import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import main

if __name__ == "__main__":
    # HGNN+ (HGNNP, attention) baseline on Cora. Own GPU via launch_cora_baselines.sh.
    main(sys.argv[1:] + ["--model", "hgnnp", "--dataset", "cora"])
