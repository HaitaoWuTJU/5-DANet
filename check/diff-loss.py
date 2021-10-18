from reprod_log import ReprodDiffHelper
import numpy as np
if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("./loss_1.npy")
    info2 = diff_helper.load_info("./loss_2.npy")

    diff_helper.compare_info(info1,info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff-loss.txt")